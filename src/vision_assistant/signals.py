from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

import cv2
import requests

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp_python = None
    mp_vision = None


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(slots=True)
class HandObservation:
    handedness: str
    finger_count: int
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class SceneSignals:
    hand_observations: list[HandObservation]
    phone_call_detected: bool
    eyes_closed_detected: bool
    drowsy_detected: bool
    notes: list[str]


@dataclass(slots=True)
class FaceObservation:
    bbox: tuple[int, int, int, int]
    eyes_detected: int


@dataclass(slots=True)
class FaceSignalState:
    closed_eye_started_at: float | None = None
    last_seen_at: float = 0.0
    closed_eye_duration: float = 0.0


class HandSignalDetector:
    def __init__(
        self,
        model_path: str | Path,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
    ):
        self.available = False
        self.landmarker = None
        if mp is not None and mp_python is not None and mp_vision is not None:
            asset_path = ensure_hand_model(model_path)
            base_options = mp_python.BaseOptions(model_asset_path=str(asset_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.45,
                min_hand_presence_confidence=0.4,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self.available = True

    def detect(self, frame_bgr) -> list[HandObservation]:
        if not self.available or self.landmarker is None or mp is None:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks or not result.handedness:
            return []

        height, width = frame_bgr.shape[:2]
        observations: list[HandObservation] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            xs = [int(point.x * width) for point in landmarks]
            ys = [int(point.y * height) for point in landmarks]
            x1, x2 = max(0, min(xs)), min(width, max(xs))
            y1, y2 = max(0, min(ys)), min(height, max(ys))
            label = handedness[0].category_name.lower()
            finger_count = count_extended_fingers(landmarks, label)
            observations.append(
                HandObservation(
                    handedness=label,
                    finger_count=finger_count,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return observations


def ensure_hand_model(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(HAND_LANDMARKER_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


class FaceSignalDetector:
    def __init__(self):
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        self.available = not self.face_cascade.empty() and not self.eye_cascade.empty()
        self.state = FaceSignalState()

    def detect(self, frame_bgr) -> tuple[list[FaceObservation], bool, bool]:
        if not self.available:
            return [], False, False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        observations: list[FaceObservation] = []
        eyes_closed_detected = False
        now = time()
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(12, 12),
            )
            eye_count = len(eyes)
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    eyes_detected=eye_count,
                )
            )
            if eye_count == 0:
                eyes_closed_detected = True

        drowsy_detected = self._update_state(len(faces) > 0, eyes_closed_detected, now)
        return observations, eyes_closed_detected, drowsy_detected

    def _update_state(
        self, face_visible: bool, eyes_closed_detected: bool, now: float
    ) -> bool:
        if not face_visible:
            self.state.closed_eye_started_at = None
            self.state.closed_eye_duration = 0.0
            return False

        self.state.last_seen_at = now
        if eyes_closed_detected:
            if self.state.closed_eye_started_at is None:
                self.state.closed_eye_started_at = now
            self.state.closed_eye_duration = now - self.state.closed_eye_started_at
        else:
            self.state.closed_eye_started_at = None
            self.state.closed_eye_duration = 0.0

        return self.state.closed_eye_duration >= 1.5


def count_extended_fingers(landmarks, handedness: str) -> int:
    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]

    count = 0
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness == "right":
        thumb_extended = thumb_tip.x < thumb_ip.x
    else:
        thumb_extended = thumb_tip.x > thumb_ip.x
    if thumb_extended:
        count += 1

    for tip_idx, pip_idx in zip(tip_indices, pip_indices):
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            count += 1
    return count


def infer_phone_call(detections: list[dict]) -> bool:
    persons = [item for item in detections if item["label"] == "person"]
    phones = [item for item in detections if item["label"] == "cell phone"]
    if not persons or not phones:
        return False

    for person in persons:
        px1, py1, px2, py2 = person["xyxy"]
        person_width = max(1.0, px2 - px1)
        person_height = max(1.0, py2 - py1)
        upper_limit = py1 + (person_height * 0.45)
        head_center_x = px1 + (person_width * 0.5)
        for phone in phones:
            x1, y1, x2, y2 = phone["xyxy"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            near_head_x = abs(cx - head_center_x) <= person_width * 0.35
            near_head_y = py1 <= cy <= upper_limit
            if near_head_x and near_head_y:
                return True
    return False


def summarize_visible_objects(detections: list[dict], max_items: int = 5) -> str:
    counts: dict[str, int] = {}
    for item in detections:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    parts = [f"{count} {label}" if count > 1 else label for label, count in ordered]
    return ", ".join(parts)


def build_scene_signals(
    hand_observations: list[HandObservation],
    detections: list[dict],
    face_observations: list[FaceObservation],
    eyes_closed_detected: bool,
    drowsy_detected: bool,
) -> SceneSignals:
    notes: list[str] = []
    if hand_observations:
        counts = ", ".join(
            f"mano {hand.handedness}: {hand.finger_count} dedos"
            for hand in hand_observations
        )
        notes.append(counts)
        if any(hand.finger_count >= 1 for hand in hand_observations):
            notes.append("hay gestos manuales visibles")

    phone_call_detected = infer_phone_call(detections)
    if phone_call_detected:
        notes.append("posible persona hablando por telefono")

    if face_observations:
        if eyes_closed_detected:
            notes.append("ojos posiblemente cerrados")
        if drowsy_detected:
            notes.append("persona con senales de somnolencia")

    visible_objects = summarize_visible_objects(detections)
    if visible_objects:
        notes.append(f"objetos visibles: {visible_objects}")

    return SceneSignals(
        hand_observations=hand_observations,
        phone_call_detected=phone_call_detected,
        eyes_closed_detected=eyes_closed_detected,
        drowsy_detected=drowsy_detected,
        notes=notes,
    )
