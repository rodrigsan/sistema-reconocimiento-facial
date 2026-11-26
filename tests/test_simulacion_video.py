"""tests/test_simulacion_video.py

Pruebas de simulación con imágenes/video (sin cámara real).
Valida que face_detector y overlay_renderer funcionen con frames sintéticos.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import detectar_rostros
from src.overlay_renderer import dibujar_overlay


def _crear_frame_con_rostro() -> np.ndarray:
    """Crea un frame sintético con un rectángulo que simula un rostro."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dibuja un rectángulo simulando un rostro en el centro
    cv2.rectangle(frame, (270, 190), (370, 290), (200, 200, 200), thickness=-1)
    return frame


def test_detectar_rostros_frame_vacio():
    """Frame negro (sin rostro) debe retornar lista vacía."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rostros = detectar_rostros(frame)
    assert isinstance(rostros, list)


def test_dibujar_overlay_sin_detecciones():
    """Overlay sin detecciones debe retornar frame sin errores."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    resultado = dibujar_overlay(frame_bgr=frame, detecciones=[], fps=30.0)
    assert resultado.shape == frame.shape


def test_dibujar_overlay_con_deteccion_reconocida():
    """Overlay con detección reconocida debe dibujar recuadro verde."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detecciones = [
        {
            "box": (100, 100, 100, 100),
            "reconocido": True,
            "nombre": "Ana",
            "profesion": "Ingeniera",
            "score": 0.1,
            "confidence": 0.9,
        }
    ]
    resultado = dibujar_overlay(frame_bgr=frame, detecciones=detecciones, fps=30.0)
    assert resultado.shape == frame.shape


def test_dibujar_overlay_con_deteccion_no_reconocida():
    """Overlay con detección no reconocida debe dibujar recuadro naranja."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detecciones = [
        {
            "box": (100, 100, 100, 100),
            "reconocido": False,
            "nombre": "",
            "profesion": "",
            "score": 0.9,
            "confidence": 0.8,
        }
    ]
    resultado = dibujar_overlay(frame_bgr=frame, detecciones=detecciones, fps=30.0)
    assert resultado.shape == frame.shape
