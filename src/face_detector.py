"""src/face_detector.py

Detección de rostros en un frame usando OpenCV (Haar cascades).

Contrato:
    detectar_rostros(frame_bgr: np.ndarray) -> list[dict]

Cada dict devuelto tiene la forma:
    {"box": (x, y, w, h), "crop": rostro_bgr, "confidence": float}
"""

from __future__ import annotations

import cv2
import numpy as np


# Carga el clasificador Haar de OpenCV (incluido en la instalación).
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detectar_rostros(frame_bgr: np.ndarray) -> list[dict]:
    """Detecta todos los rostros en un frame BGR.

    Args:
        frame_bgr: Imagen en formato BGR (numpy array).

    Returns:
        Lista de dicts, uno por rostro detectado, cada uno con:
            - "box": (x, y, w, h) coordenadas del rectángulo.
            - "crop": recorte BGR del rostro.
            - "confidence": puntuación de confianza (0.0 - 1.0).
    """
    if frame_bgr is None:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # detectMultiScale devuelve rectángulos (x, y, w, h).
    # scaleFactor y minNeighbors son parámetros clave para equilibrio
    # entre velocidad y precisión.
    rostros = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    resultados: list[dict] = []
    for (x, y, w, h) in rostros:
        # Recorte del rostro con un pequeño margen para incluir contexto.
        margen = int(0.15 * min(w, h))
        x0 = max(0, x - margen)
        y0 = max(0, y - margen)
        x1 = min(frame_bgr.shape[1], x + w + margen)
        y1 = min(frame_bgr.shape[0], y + h + margen)
        crop = frame_bgr[y0:y1, x0:x1]

        # Confianza aproximada basada en el área del rostro detectado.
        # (Haar no devuelve puntuación directa; usamos un valor heurístico.)
        area = w * h
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        confianza = min(1.0, area / (frame_area * 0.25))

        resultados.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "crop": crop,
                "confidence": float(confianza),
            }
        )

    return resultados
