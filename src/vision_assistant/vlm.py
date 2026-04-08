from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol

import cv2
import requests


def frame_to_base64(frame, max_edge: int = 448, jpeg_quality: int = 55) -> str:
    height, width = frame.shape[:2]
    longest_edge = max(height, width)
    if longest_edge > max_edge:
        scale = max_edge / float(longest_edge)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        frame = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not ok:
        raise RuntimeError("No se pudo codificar el frame para el VLM")
    return base64.b64encode(buffer).decode("utf-8")


class VisionLLM(Protocol):
    def describe_scene(
        self,
        frame,
        object_labels: list[str],
        context_name: str,
        recent_events: list[str],
    ) -> str: ...

    def healthcheck(self) -> tuple[bool, str]: ...


@dataclass(slots=True)
class LMStudioVisionLLM:
    model: str
    base_url: str
    timeout: int = 60
    image_max_edge: int = 448
    jpeg_quality: int = 55
    max_labels: int = 8

    def describe_scene(
        self,
        frame,
        object_labels: list[str],
        context_name: str,
        recent_events: list[str],
    ) -> str:
        image_b64 = frame_to_base64(
            frame, max_edge=self.image_max_edge, jpeg_quality=self.jpeg_quality
        )
        labels = ", ".join(object_labels[: self.max_labels]) or "none"
        events = "; ".join(recent_events[:4]) or "sin eventos recientes"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 60,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente visual local rapido. "
                        "Responde en espanol con una sola frase corta de menos de 20 palabras. "
                        "Menciona solo lo mas visible o la accion principal. "
                        "Sin listas. Sin especular."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Contexto estimado: {context_name}. "
                                f"Etiquetas detectadas: {labels}. "
                                f"Eventos recientes: {events}. "
                                "Describe solo la escena actual."
                            ),
                        },
                    ],
                },
            ],
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/v1/models", timeout=10
            )
            response.raise_for_status()
            return True, "LM Studio disponible"
        except Exception as exc:
            return False, f"LM Studio no disponible: {exc}"


@dataclass(slots=True)
class OllamaVisionLLM:
    model: str
    base_url: str
    timeout: int = 60
    image_max_edge: int = 448
    jpeg_quality: int = 55
    max_labels: int = 8

    def describe_scene(
        self,
        frame,
        object_labels: list[str],
        context_name: str,
        recent_events: list[str],
    ) -> str:
        image_b64 = frame_to_base64(
            frame, max_edge=self.image_max_edge, jpeg_quality=self.jpeg_quality
        )
        labels = ", ".join(object_labels[: self.max_labels]) or "none"
        events = "; ".join(recent_events[:4]) or "sin eventos recientes"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente visual local rapido. "
                        "Responde en espanol con una sola frase corta de menos de 20 palabras. "
                        "Menciona solo lo mas visible o la accion principal. "
                        "Sin listas. Sin especular."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Contexto estimado: {context_name}. "
                        f"Etiquetas detectadas: {labels}. "
                        f"Eventos recientes: {events}. "
                        "Describe solo la escena actual."
                    ),
                    "images": [image_b64],
                },
            ],
            "options": {"temperature": 0.1, "num_predict": 60},
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=10)
            response.raise_for_status()
            return True, "Ollama disponible"
        except Exception as exc:
            return False, f"Ollama no disponible: {exc}"


def build_vlm(
    runtime: str,
    model: str,
    lmstudio_base_url: str,
    ollama_base_url: str,
    image_max_edge: int,
    jpeg_quality: int,
    max_labels: int,
) -> VisionLLM:
    runtime_normalized = runtime.strip().lower()
    if runtime_normalized == "lmstudio":
        return LMStudioVisionLLM(
            model=model,
            base_url=lmstudio_base_url,
            image_max_edge=image_max_edge,
            jpeg_quality=jpeg_quality,
            max_labels=max_labels,
        )
    if runtime_normalized == "ollama":
        return OllamaVisionLLM(
            model=model,
            base_url=ollama_base_url,
            image_max_edge=image_max_edge,
            jpeg_quality=jpeg_quality,
            max_labels=max_labels,
        )
    raise ValueError(f"Runtime VLM no soportado: {runtime}")
