"""src/overlay_renderer.py

Renderizado de recuadros, etiquetas y FPS sobre el frame.

Contrato:
    dibujar_overlay(frame_bgr: np.ndarray, detecciones: list[dict],
                    fps: float) -> np.ndarray

Colores (BGR):
    - Reconocido: Verde (0, 200, 80)
    - No reconocido: Naranja (0, 165, 255) con texto
      "Persona no existente en la base de datos"
"""

from __future__ import annotations

import cv2
import numpy as np

# Colores BGR
COLOR_RECONOCIDO = (0, 200, 80)      # Verde
COLOR_NO_RECONOCIDO = (0, 165, 255)  # Naranja
COLOR_TEXTO = (255, 255, 255)        # Blanco
COLOR_FONDO_TEXTO = (0, 0, 0)        # Negro para fondo de etiqueta

TEXTO_NO_REGISTRADO = "Persona no existente en la base de datos"


def _dibujar_texto_con_fondo(
    frame: np.ndarray,
    texto: str,
    origen: tuple[int, int],
    color_texto: tuple[int, int, int] = COLOR_TEXTO,
    color_fondo: tuple[int, int, int] = COLOR_FONDO_TEXTO,
    escala: float = 0.6,
    grosor: int = 2,
) -> None:
    """Dibuja texto con rectángulo de fondo para legibilidad."""
    (ancho_texto, alto_texto), linea_base = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor
    )
    x, y = origen
    x0 = max(0, x)
    y0 = max(alto_texto + 5, y)
    x1 = min(frame.shape[1], x + ancho_texto + 6)
    y1 = min(frame.shape[0], y + linea_base + 6)

    cv2.rectangle(frame, (x0, y0 - alto_texto - 5), (x1, y1), color_fondo, thickness=cv2.FILLED)
    cv2.putText(
        frame,
        texto,
        (x0 + 3, y0 - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color_texto,
        grosor,
        lineType=cv2.LINE_AA,
    )


def dibujar_overlay(
    frame_bgr: np.ndarray,
    detecciones: list[dict],
    fps: float,
) -> np.ndarray:
    """Dibuja recuadros, etiquetas y FPS sobre el frame.

    Args:
        frame_bgr: Frame original en BGR.
        detecciones: Lista de dicts con claves:
            - "box": (x, y, w, h)
            - "reconocido": bool
            - "nombre": str
            - "profesion": str
            - "score": float
        fps: FPS actual para mostrar en esquina.

    Returns:
        Frame con overlay dibujado (misma imagen modificada).
    """
    frame = frame_bgr.copy()

    # Dibujar FPS en esquina superior izquierda
    texto_fps = f"FPS: {fps:.1f}"
    _dibujar_texto_con_fondo(
        frame,
        texto_fps,
        (10, 30),
        color_texto=(0, 255, 0),
        escala=0.7,
        grosor=2,
    )

    for det in detecciones:
        x, y, w, h = det["box"]
        x1 = x + w
        y1 = y + h

        reconocido = det.get("reconocido", False)
        nombre = det.get("nombre", "")
        profesion = det.get("profesion", "")
        score = det.get("score", 0.0)

        if reconocido:
            color = COLOR_RECONOCIDO
            etiqueta_linea1 = nombre
            etiqueta_linea2 = profesion
        else:
            color = COLOR_NO_RECONOCIDO
            etiqueta_linea1 = TEXTO_NO_REGISTRADO
            etiqueta_linea2 = f"(score: {score:.2f})"

        # Recuadro alrededor del rostro
        cv2.rectangle(frame, (x, y), (x1, y1), color, thickness=3)

        # Etiqueta de nombre y profesión debajo del recuadro
        texto_y = y1 + 25
        _dibujar_texto_con_fondo(frame, etiqueta_linea1, (x, texto_y), escala=0.6, grosor=2)

        if reconocido:
            texto_y2 = texto_y + 30
            _dibujar_texto_con_fondo(
                frame, etiqueta_linea2, (x, texto_y2), escala=0.5, grosor=1
            )

    return frame
