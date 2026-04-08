from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from time import time


DESK_OBJECTS = {
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "cell phone",
    "cup",
    "bottle",
    "tv",
}

PROCEDURE_OBJECTS = {
    "scissors",
    "knife",
    "bottle",
    "cup",
    "book",
    "cell phone",
}

ACTIVITY_OBJECTS = {
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
}


@dataclass(slots=True)
class DetectionRecord:
    track_id: int
    label: str
    confidence: float
    center_x: float
    center_y: float
    area_ratio: float


@dataclass(slots=True)
class TrackedEntity:
    track_id: int
    label: str
    first_seen_at: float
    last_seen_at: float
    seen_frames: int = 1
    last_center_x: float = 0.0
    last_center_y: float = 0.0
    movement_score: float = 0.0
    area_ratio: float = 0.0


@dataclass(slots=True)
class Event:
    timestamp: float
    kind: str
    message: str


@dataclass(slots=True)
class ContextSnapshot:
    name: str
    confidence: float
    rationale: str
    desk_score: float
    activity_score: float
    procedure_score: float


@dataclass(slots=True)
class SceneMemory:
    entities: dict[int, TrackedEntity] = field(default_factory=dict)
    recent_events: deque[Event] = field(default_factory=lambda: deque(maxlen=12))
    recent_labels: deque[list[str]] = field(default_factory=lambda: deque(maxlen=20))
    last_context: ContextSnapshot = field(
        default_factory=lambda: ContextSnapshot(
            name="indeterminado",
            confidence=0.0,
            rationale="Sin suficientes datos",
            desk_score=0.0,
            activity_score=0.0,
            procedure_score=0.0,
        )
    )
    last_event_at: dict[str, float] = field(default_factory=dict)

    def update(self, detections: list[DetectionRecord]) -> ContextSnapshot:
        now = time()
        labels = sorted({d.label for d in detections})
        self.recent_labels.append(labels)

        active_track_ids: set[int] = set()
        for detection in detections:
            active_track_ids.add(detection.track_id)
            entity = self.entities.get(detection.track_id)
            if entity is None:
                entity = TrackedEntity(
                    track_id=detection.track_id,
                    label=detection.label,
                    first_seen_at=now,
                    last_seen_at=now,
                    last_center_x=detection.center_x,
                    last_center_y=detection.center_y,
                    area_ratio=detection.area_ratio,
                )
                self.entities[detection.track_id] = entity
                self._add_event(
                    kind=f"new:{detection.label}",
                    message=f"Nuevo {detection.label} detectado",
                    min_interval=4.0,
                )
            else:
                dx = detection.center_x - entity.last_center_x
                dy = detection.center_y - entity.last_center_y
                entity.movement_score += abs(dx) + abs(dy)
                entity.last_seen_at = now
                entity.seen_frames += 1
                entity.last_center_x = detection.center_x
                entity.last_center_y = detection.center_y
                entity.area_ratio = detection.area_ratio

        expired_track_ids = [
            track_id
            for track_id, entity in self.entities.items()
            if track_id not in active_track_ids and now - entity.last_seen_at > 2.5
        ]
        for track_id in expired_track_ids:
            entity = self.entities.pop(track_id)
            self._add_event(
                kind=f"gone:{entity.label}",
                message=f"{entity.label} salio de escena",
                min_interval=4.0,
            )

        self.last_context = infer_context(self.entities, list(self.recent_events), labels)
        return self.last_context

    def _add_event(self, kind: str, message: str, min_interval: float) -> None:
        now = time()
        if now - self.last_event_at.get(kind, 0.0) < min_interval:
            return
        self.recent_events.appendleft(Event(timestamp=now, kind=kind, message=message))
        self.last_event_at[kind] = now

    def summarize_events(self, limit: int = 4) -> list[str]:
        return [event.message for event in list(self.recent_events)[:limit]]


def infer_context(
    entities: dict[int, TrackedEntity], recent_events: list[Event], current_labels: list[str]
) -> ContextSnapshot:
    label_counter = Counter(entity.label for entity in entities.values())
    desk_score = sum(label_counter[label] for label in DESK_OBJECTS)
    activity_score = sum(label_counter[label] for label in ACTIVITY_OBJECTS)
    procedure_score = sum(label_counter[label] for label in PROCEDURE_OBJECTS)

    person_count = label_counter.get("person", 0)
    if person_count:
        activity_score += 1.5 * person_count

    moved_objects = sum(1 for entity in entities.values() if entity.movement_score > 0.22)
    stable_objects = sum(1 for entity in entities.values() if entity.seen_frames > 30)
    if moved_objects >= 2:
        procedure_score += 2.0
    if stable_objects >= 3 and {"laptop", "keyboard"}.intersection(current_labels):
        desk_score += 2.0
    if len(recent_events) >= 2 and person_count:
        activity_score += 1.0

    scoreboard = {
        "escritorio": desk_score,
        "actividad": activity_score,
        "procedimiento": procedure_score,
    }
    context_name, top_score = max(scoreboard.items(), key=lambda item: item[1])
    total_score = sum(scoreboard.values())
    confidence = top_score / total_score if total_score > 0 else 0.0
    if top_score <= 1.0:
        context_name = "indeterminado"
        confidence = 0.0

    rationale = build_rationale(context_name, current_labels, moved_objects, person_count)
    return ContextSnapshot(
        name=context_name,
        confidence=confidence,
        rationale=rationale,
        desk_score=desk_score,
        activity_score=activity_score,
        procedure_score=procedure_score,
    )


def build_rationale(
    context_name: str, current_labels: list[str], moved_objects: int, person_count: int
) -> str:
    if context_name == "escritorio":
        return "Objetos tipicos de escritorio estables en escena"
    if context_name == "actividad":
        return f"Presencia de {person_count} persona(s) y dinamica de entrada/salida"
    if context_name == "procedimiento":
        return f"Hay manipulacion de objetos y cambios frecuentes ({moved_objects})"
    if current_labels:
        return f"Escena mixta: {', '.join(current_labels[:4])}"
    return "Sin suficientes datos visuales"
