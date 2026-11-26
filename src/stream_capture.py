"""src/stream_capture.py

Captura de video desde stream IP/HTTP de celular (IP Webcam/DroidCam)
o webcam local.

Contratos (según README del proyecto):
    abrir_stream(fuente: str) -> cv2.VideoCapture
    leer_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


def abrir_stream(fuente: str) -> cv2.VideoCapture:
    """Abre una URL RTSP/HTTP o el índice de cámara local.

    Args:
        fuente: URL del stream (ej. "http://192.168.43.1:8080/video")
                o índice de cámara (ej. "0" para webcam local).

    Returns:
        Objeto cv2.VideoCapture abierto.
    """
    # Si el fuente es un string numérico, lo tratamos como índice de cámara.
    if fuente.isdigit():
        cap = cv2.VideoCapture(int(fuente))
    else:
        cap = cv2.VideoCapture(fuente)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    return cap


def leer_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray | None]:
    """Lee un frame del stream.

    Args:
        cap: Objeto cv2.VideoCapture abierto.

    Returns:
        Tupla (ok, frame_bgr) donde ok indica si la lectura fue exitosa
        y frame_bgr es el frame en formato BGR (o None si falló).
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    return True, frame
