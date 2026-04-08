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


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Vision Assistant Local")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--detector-model", default="yolo11s.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.4)
    parser.add_argument("--vlm-runtime", choices=["lmstudio", "ollama"], default="lmstudio")
    parser.add_argument("--vlm-model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--lmstudio-base-url", default="http://localhost:1234")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--scene-cooldown", type=float, default=6.0)
    parser.add_argument("--scene-change-min-interval", type=float, default=2.5)
    parser.add_argument("--vlm-image-max-edge", type=int, default=448)
    parser.add_argument("--vlm-jpeg-quality", type=int, default=55)
    parser.add_argument("--vlm-max-labels", type=int, default=8)
    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera_index,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
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
    )


def start_vlm_worker(config: AppConfig, state: AssistantState):
    task_queue: queue.Queue[tuple[object, list[str], str, str, list[str]]] = queue.Queue(maxsize=1)
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
            frame, labels, reason, context_name, recent_events = task_queue.get()
            started = time.time()
            state.is_describing = True
            try:
                description = vlm.describe_scene(frame, labels, context_name, recent_events)
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


def draw_overlay(frame, fps: float, detected_count: int, runtime_label: str, state: AssistantState) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 110), (15, 15, 15), -1)
    cv2.rectangle(overlay, (0, height - 120), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "VISION ASSISTANT LOCAL", (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objetos: {detected_count}", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Runtime: {runtime_label}", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"VLM: {state.vlm_status}", (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    activity = "Analizando..." if state.is_describing else f"Latencia VLM: {state.last_vlm_latency_ms:.0f} ms"
    cv2.putText(frame, activity, (width - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if state.latest_context is not None:
        context_label = f"Contexto: {state.latest_context.name} ({state.latest_context.confidence:.0%})"
        cv2.putText(frame, context_label, (width - 320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 220, 255), 1, cv2.LINE_AA)

    description = fit_text(state.latest_description, 220)
    cv2.putText(frame, "Asistente:", (16, height - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, description, (16, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q salir  P pausa  S screenshot  D describir", (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    if state.recent_events:
        event_line = fit_text(" | ".join(state.recent_events[:2]), 120)
        cv2.putText(frame, f"Eventos: {event_line}", (360, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def queue_scene_description(
    task_queue,
    frame,
    labels: list[str],
    reason: str,
    context_name: str,
    recent_events: list[str],
) -> bool:
    if task_queue.full():
        return False
    task_queue.put_nowait((frame.copy(), labels, reason, context_name, recent_events))
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
    conf_values = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else [0.0] * len(cls_values)
    if boxes.id is not None:
        track_values = boxes.id.detach().cpu().tolist()
    else:
        track_values = list(range(len(cls_values)))

    detections: list[DetectionRecord] = []
    for cls_idx, xyxy, conf, track_id in zip(cls_values, xyxy_values, conf_values, track_values):
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


def main() -> None:
    config = parse_args()
    config.screenshot_dir.mkdir(exist_ok=True)

    state = AssistantState()
    state.recent_events = []
    task_queue = start_vlm_worker(config, state)
    detector = YOLO(config.detector_model)
    scene_memory = SceneMemory()

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.cam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {config.camera_index}")

    print("Vision Assistant Local")
    print("Q=salir P=pausa S=screenshot D=descripcion")

    paused = False
    last_fps_update = time.time()
    frames_since_update = 0
    fps = 0.0
    current_frame = None
    unique_labels: list[str] = []
    frame = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frames_since_update += 1
            elapsed = time.time() - last_fps_update
            if elapsed >= 0.5:
                fps = frames_since_update / elapsed
                frames_since_update = 0
                last_fps_update = time.time()

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
            labels = [detection.label for detection in detections]
            unique_labels = sorted(set(labels))
            context = scene_memory.update(detections)
            recent_events = scene_memory.summarize_events(limit=config.max_recent_events)
            state.latest_context = context
            state.recent_events = recent_events

            now = time.time()
            scene_signature = build_scene_signature(unique_labels + [context.name])
            should_describe_by_change = (
                scene_signature
                and scene_signature != state.last_scene_signature
                and now - state.last_scene_queued_at >= config.scene_change_min_interval_seconds
            )
            should_describe_by_cooldown = (
                unique_labels
                and now - state.last_description_at >= config.scene_cooldown_seconds
                and now - state.last_scene_queued_at >= config.scene_change_min_interval_seconds
            )
            if state.vlm_available and not state.is_describing and (should_describe_by_change or should_describe_by_cooldown):
                if queue_scene_description(
                    task_queue,
                    frame,
                    unique_labels,
                    "scene",
                    context.name,
                    recent_events,
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

        if current_frame is None:
            continue

        cv2.imshow("Vision Assistant Local", current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("s"):
            filename = config.screenshot_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), current_frame)
            print(f"Screenshot guardado en {filename}")
        if key == ord("d") and not paused and state.vlm_available and frame is not None:
            context_name = state.latest_context.name if state.latest_context is not None else "indeterminado"
            if queue_scene_description(
                task_queue,
                frame,
                unique_labels,
                "manual",
                context_name,
                state.recent_events or [],
            ):
                state.last_scene_queued_at = time.time()

    cap.release()
    cv2.destroyAllWindows()
