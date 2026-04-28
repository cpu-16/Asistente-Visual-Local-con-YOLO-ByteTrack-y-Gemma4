from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from vision_assistant.config import AppConfig
from vision_assistant.context import ContextSnapshot, DetectionRecord, SceneMemory
from vision_assistant.signals import (
    FaceSignalDetector,
    HandSignalDetector,
    build_scene_signals,
)
from vision_assistant.vlm import build_vlm


@dataclass(slots=True)
class AssistantState:
    latest_description: str = "Esperando descripcion..."
    last_description_at: float = 0.0
    last_vlm_latency_ms: float = 0.0
    vlm_available: bool = False
    vlm_status: str = "Sin verificar"
    last_error: str = ""
    is_describing: bool = False
    last_scene_signature: str = ""
    last_scene_queued_at: float = 0.0
    latest_context: ContextSnapshot | None = None
    recent_events: list[str] = None
    latest_scene_notes: list[str] = None
    capture_status: str = "Camara lista"


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Vision Assistant Local")
    parser.add_argument(
        "--camera-source",
        default="0",
        help="Indice de camara local o URL de stream (http/rtsp)",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--display-scale", type=float, default=1.15)
    parser.add_argument("--camera-open-timeout-ms", type=int, default=8000)
    parser.add_argument("--camera-read-timeout-ms", type=int, default=12000)
    parser.add_argument("--camera-reconnect-delay", type=float, default=2.0)
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    parser.add_argument("--detector-model", default="yolo11s.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.4)
    parser.add_argument(
        "--vlm-runtime", choices=["lmstudio", "ollama"], default="lmstudio"
    )
    parser.add_argument("--vlm-model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--scene-cooldown", type=float, default=6.0)
    parser.add_argument("--scene-change-min-interval", type=float, default=2.5)
    parser.add_argument("--vlm-image-max-edge", type=int, default=448)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=55)
    parser.add_argument("--vlm-max-labels", type=int, default=8)
    parser.add_argument("--disable-hand-signals", action="store_true")
    args = parser.parse_args()
    return AppConfig(
        camera_source=args.camera_source,
        camera_index=args.camera_index,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        display_scale=max(0.5, args.display_scale),
        camera_open_timeout_ms=args.camera_open_timeout_ms,
        camera_read_timeout_ms=args.camera_read_timeout_ms,
        camera_reconnect_delay_seconds=args.camera_reconnect_delay,
        process_every_n_frames=max(1, args.process_every_n_frames),
        detector_model=args.detector_model,
        detector_confidence=args.detector_confidence,
        vlm_runtime=args.vlm_runtime,
        vlm_model=args.vlm_model,
        lmstudio_base_url=args.lmstudio_base_url,
        ollama_base_url=args.ollama_base_url,
        scene_cooldown_seconds=args.scene_cooldown,
        scene_change_min_interval_seconds=args.scene_change_min_interval,
        vlm_image_max_edge=args.vlm_image_max_edge,
        vlm_jpeg_quality=args.vlm_jpeg_quality,
        vlm_max_labels=args.vlm_max_labels,
        enable_hand_signals=not args.disable_hand_signals,
    )


def build_video_capture(config: AppConfig) -> cv2.VideoCapture:
    source = config.camera_source.strip()
    is_stream = "://" in source
    if source.isdigit():
        source_value: int | str = int(source)
    else:
        source_value = source

    cap = cv2.VideoCapture(source_value)
    if is_stream:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.camera_open_timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.camera_read_timeout_ms)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def is_stream_source(camera_source: str) -> bool:
    return "://" in camera_source


def reconnect_video_capture(config: AppConfig, state: AssistantState):
    state.capture_status = "Reconectando stream..."
    time.sleep(config.camera_reconnect_delay_seconds)
    cap = build_video_capture(config)
    if cap.isOpened():
        state.capture_status = "Stream reconectado"
        return cap
    state.capture_status = "Esperando stream del celular"
    return None


def start_vlm_worker(config: AppConfig, state: AssistantState):
    task_queue: queue.Queue[
        tuple[object, list[str], str, str, list[str], list[str]]
    ] = queue.Queue(maxsize=1)
    vlm = build_vlm(
        runtime=config.vlm_runtime,
        model=config.vlm_model,
        lmstudio_base_url=config.lmstudio_base_url,
        ollama_base_url=config.ollama_base_url,
        image_max_edge=config.vlm_image_max_edge,
        jpeg_quality=config.vlm_jpeg_quality,
        max_labels=config.vlm_max_labels,
    )

    ok, message = vlm.healthcheck()
    state.vlm_available = ok
    state.vlm_status = message

    def worker() -> None:
        while True:
            frame, labels, reason, context_name, recent_events, scene_notes = (
                task_queue.get()
            )
            started = time.time()
            state.is_describing = True
            try:
                description = vlm.describe_scene(
                    frame, labels, context_name, recent_events, scene_notes
                )
                state.latest_description = description
                state.last_error = ""
            except Exception as exc:
                state.last_error = str(exc)
                state.latest_description = f"Error VLM ({reason}): {exc}"
            finally:
                state.last_vlm_latency_ms = (time.time() - started) * 1000
                state.last_description_at = time.time()
                state.is_describing = False
                task_queue.task_done()

    threading.Thread(target=worker, daemon=True).start()
    return task_queue


def fit_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def wrap_text_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    words = " ".join(text.split()).split()
    if not words:
        return [""]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if len(lines) < max_lines and current:
        lines.append(current)

    if len(lines) == max_lines and words:
        remaining_joined = " ".join(words)
        if " ".join(lines) != remaining_joined:
            lines[-1] = fit_text(lines[-1] + " " + " ".join(words[len(" ".join(lines).split()):]), max_chars)
    return lines[:max_lines]


def draw_overlay(
    frame, fps: float, detected_count: int, runtime_label: str, state: AssistantState
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 110), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, height - 170), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame,
        "VISION ASSISTANT LOCAL",
        (16, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 220, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Objetos: {detected_count}",
        (130, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Runtime: {runtime_label}",
        (260, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"VLM: {state.vlm_status}",
        (16, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    activity = (
        "Analizando..."
        if state.is_describing
        else f"Latencia VLM: {state.last_vlm_latency_ms:.0f} ms"
    )
    cv2.putText(
        frame,
        activity,
        (width - 220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if state.latest_context is not None:
        context_label = f"Contexto: {state.latest_context.name} ({state.latest_context.confidence:.0%})"
        cv2.putText(
            frame,
            context_label,
            (width - 320, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (140, 220, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        state.capture_status,
        (width - 320, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 210, 120),
        1,
        cv2.LINE_AA,
    )

    description_lines = wrap_text_lines(state.latest_description, 95, 2)
    cv2.putText(
        frame,
        "Asistente:",
        (16, height - 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 120),
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(description_lines):
        cv2.putText(
            frame,
            line,
            (16, height - 96 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Q salir  P pausa  S screenshot  D describir",
        (16, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    if state.recent_events:
        event_line = fit_text(" | ".join(state.recent_events[:2]), 85)
        cv2.putText(
            frame,
            f"Eventos: {event_line}",
            (16, height - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    if state.latest_scene_notes:
        note_line = fit_text(" | ".join(state.latest_scene_notes[:2]), 80)
        cv2.putText(
            frame,
            f"Senales: {note_line}",
            (width // 2, height - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )


def queue_scene_description(
    task_queue,
    frame,
    labels: list[str],
    reason: str,
    context_name: str,
    recent_events: list[str],
    scene_notes: list[str],
) -> bool:
    if task_queue.full():
        return False
    task_queue.put_nowait(
        (frame.copy(), labels, reason, context_name, recent_events, scene_notes)
    )
    return True


def build_scene_signature(labels: list[str]) -> str:
    return "|".join(labels)


def extract_detections(result, frame_shape) -> list[DetectionRecord]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None or boxes.xyxy is None:
        return []

    height, width = frame_shape[:2]
    cls_values = boxes.cls.detach().cpu().tolist()
    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    conf_values = (
        boxes.conf.detach().cpu().tolist()
        if boxes.conf is not None
        else [0.0] * len(cls_values)
    )
    if boxes.id is not None:
        track_values = boxes.id.detach().cpu().tolist()
    else:
        track_values = list(range(len(cls_values)))

    detections: list[DetectionRecord] = []
    for cls_idx, xyxy, conf, track_id in zip(
        cls_values, xyxy_values, conf_values, track_values
    ):
        x1, y1, x2, y2 = xyxy
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        detections.append(
            DetectionRecord(
                track_id=int(track_id),
                label=result.names.get(int(cls_idx), str(int(cls_idx))),
                confidence=float(conf),
                center_x=((x1 + x2) / 2.0) / width,
                center_y=((y1 + y2) / 2.0) / height,
                area_ratio=(box_width * box_height) / float(width * height),
            )
        )
    return detections


def extract_detection_summary(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None or boxes.xyxy is None:
        return []

    cls_values = boxes.cls.detach().cpu().tolist()
    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    summaries: list[dict] = []
    for cls_idx, xyxy in zip(cls_values, xyxy_values):
        summaries.append(
            {
                "label": result.names.get(int(cls_idx), str(int(cls_idx))),
                "xyxy": tuple(float(value) for value in xyxy),
            }
        )
    return summaries


def main() -> None:
    config = parse_args()
    config.screenshot_dir.mkdir(exist_ok=True)

    state = AssistantState()
    state.recent_events = []
    state.latest_scene_notes = []
    task_queue = start_vlm_worker(config, state)
    detector = YOLO(config.detector_model)
    scene_memory = SceneMemory()
    hand_signal_detector = (
        HandSignalDetector(model_path=config.hand_model_path)
        if config.enable_hand_signals
        else None
    )
    face_signal_detector = FaceSignalDetector()

    cap = build_video_capture(config)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de camara {config.camera_source}"
        )

    print("Vision Assistant Local")
    print("Q=salir P=pausa S=screenshot D=descripcion")
    cv2.namedWindow("Vision Assistant Local", cv2.WINDOW_NORMAL)

    paused = False
    last_fps_update = time.time()
    frames_since_update = 0
    fps = 0.0
    current_frame = None
    unique_labels: list[str] = []
    frame = None
    stream_source = is_stream_source(config.camera_source)
    processed_frame_index = 0
    last_processed_frame = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if stream_source:
                    cap.release()
                    new_cap = reconnect_video_capture(config, state)
                    if new_cap is None:
                        continue
                    cap = new_cap
                    continue
                break
            state.capture_status = "Camara activa"
            if not stream_source:
                frame = cv2.flip(frame, 1)
            frames_since_update += 1
            processed_frame_index += 1
            elapsed = time.time() - last_fps_update
            if elapsed >= 0.5:
                fps = frames_since_update / elapsed
                frames_since_update = 0
                last_fps_update = time.time()

            should_process_frame = (
                processed_frame_index % config.process_every_n_frames == 0
                or last_processed_frame is None
            )

            if not should_process_frame:
                preview_frame = frame.copy()
                draw_overlay(
                    preview_frame,
                    fps=fps,
                    detected_count=len(unique_labels),
                    runtime_label=config.vlm_runtime,
                    state=state,
                )
                current_frame = preview_frame
                goto_show = True
            else:
                goto_show = False

            if not goto_show:
                result = detector.track(
                    frame,
                    persist=True,
                    tracker=config.tracker_config,
                    conf=config.detector_confidence,
                    classes=config.detector_classes or None,
                    verbose=False,
                    device=0 if torch.cuda.is_available() else "cpu",
                )[0]

                annotated = result.plot(labels=True, boxes=True)
                detections = extract_detections(result, frame.shape)
                detection_summary = extract_detection_summary(result)
                labels = [detection.label for detection in detections]
                unique_labels = sorted(set(labels))
                context = scene_memory.update(detections)
                recent_events = scene_memory.summarize_events(
                    limit=config.max_recent_events
                )
                hand_observations = (
                    hand_signal_detector.detect(frame)
                    if hand_signal_detector is not None
                    else []
                )
                face_observations, eyes_closed_detected, drowsy_detected = (
                    face_signal_detector.detect(frame)
                )
                scene_signals = build_scene_signals(
                    hand_observations,
                    detection_summary,
                    face_observations,
                    eyes_closed_detected,
                    drowsy_detected,
                )
                state.latest_context = context
                state.recent_events = recent_events
                state.latest_scene_notes = scene_signals.notes

                for hand in scene_signals.hand_observations:
                    x1, y1, x2, y2 = hand.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 180, 255), 2)
                    cv2.putText(
                        annotated,
                        f"mano {hand.handedness}: {hand.finger_count}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (80, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if scene_signals.phone_call_detected:
                    cv2.putText(
                        annotated,
                        "Posible llamada telefonica",
                        (16, 116),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 200, 80),
                        2,
                        cv2.LINE_AA,
                    )

                for face in face_observations:
                    x1, y1, x2, y2 = face.bbox
                    color = (80, 220, 255)
                    if scene_signals.drowsy_detected:
                        color = (80, 80, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    face_label = (
                        "Somnolencia"
                        if scene_signals.drowsy_detected
                        else "Ojos cerrados"
                        if scene_signals.eyes_closed_detected
                        else "Rostro"
                    )
                    cv2.putText(
                        annotated,
                        face_label,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                now = time.time()
                scene_signature = build_scene_signature(
                    unique_labels + [context.name] + scene_signals.notes[:2]
                )
                should_describe_by_change = (
                    scene_signature
                    and scene_signature != state.last_scene_signature
                    and now - state.last_scene_queued_at
                    >= config.scene_change_min_interval_seconds
                )
                should_describe_by_cooldown = (
                    unique_labels
                    and now - state.last_description_at >= config.scene_cooldown_seconds
                    and now - state.last_scene_queued_at
                    >= config.scene_change_min_interval_seconds
                )
                if (
                    state.vlm_available
                    and not state.is_describing
                    and (should_describe_by_change or should_describe_by_cooldown)
                ):
                    if queue_scene_description(
                        task_queue,
                        frame,
                        unique_labels,
                        "scene",
                        context.name,
                        recent_events,
                        scene_signals.notes,
                    ):
                        state.last_scene_signature = scene_signature
                        state.last_scene_queued_at = now

                draw_overlay(
                    annotated,
                    fps=fps,
                    detected_count=len(labels),
                    runtime_label=config.vlm_runtime,
                    state=state,
                )
                current_frame = annotated
                last_processed_frame = annotated.copy()

        if current_frame is None:
            continue

        display_frame = current_frame
        if config.display_scale != 1.0:
            display_frame = cv2.resize(
                current_frame,
                None,
                fx=config.display_scale,
                fy=config.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow("Vision Assistant Local", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("s"):
            filename = (
                config.screenshot_dir
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(str(filename), current_frame)
            print(f"Screenshot guardado en {filename}")
        if key == ord("d") and not paused and state.vlm_available and frame is not None:
            context_name = (
                state.latest_context.name
                if state.latest_context is not None
                else "indeterminado"
            )
            if queue_scene_description(
                task_queue,
                frame,
                unique_labels,
                "manual",
                context_name,
                state.recent_events or [],
                state.latest_scene_notes or [],
            ):
                state.last_scene_queued_at = time.time()

    cap.release()
    cv2.destroyAllWindows()
