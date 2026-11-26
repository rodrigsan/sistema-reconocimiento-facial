"""src/display_output.py

Configuración de ventana y visualización de frames.

Contratos:
    configurar_ventana(nombre_ventana: str, fullscreen: bool = True) -> None
    mostrar_frame(nombre_ventana: str, frame_bgr: np.ndarray) -> int
"""

from __future__ import annotations

import cv2
import numpy as np


VENTANA_POR_DEFECTO = "Evento - Identificacion en Vivo"


def configurar_ventana(nombre_ventana: str, fullscreen: bool = True) -> None:
    """Configura la ventana de visualización.

    Args:
        nombre_ventana: Título de la ventana.
        fullscreen: Si True, la ventana se abre en pantalla completa.
    """
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(
            nombre_ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    else:
        cv2.setWindowProperty(
            nombre_ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
        )


def mostrar_frame(nombre_ventana: str, frame_bgr: np.ndarray) -> int:
    """Muestra un frame en la ventana y devuelve la tecla presionada.

    Args:
        nombre_ventana: Título de la ventana.
        frame_bgr: Frame en formato BGR.

    Returns:
        Código de la tecla presionada (waitKey(1)) o -1 si no hubo tecla.
    """
    cv2.imshow(nombre_ventana, frame_bgr)
    tecla = cv2.waitKey(1) & 0xFF
    return tecla
