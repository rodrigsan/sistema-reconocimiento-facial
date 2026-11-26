"""src/performance.py

Controlador de ritmo y cálculo de FPS suavizados.

Contrato:
    ControladorRitmo(cada_n_frames: int = 3)

Decide en qué frames ejecutar detección/reconocimiento y
calcula FPS suavizados para el overlay.
"""

from __future__ import annotations

import time


class ControladorRitmo:
    """Controla cada cuántos frames se ejecuta la detección pesada."""

    def __init__(self, cada_n_frames: int = 3):
        self.cada_n_frames = max(1, int(cada_n_frames))
        self._frame_counter = 0
        self._tiempo_anterior = time.perf_counter()
        self._fps_suavizado = 0.0
        self._alpha = 0.2  # factor de suavizado (0.0 - 1.0)

    def debe_procesar(self) -> bool:
        """Retorna True si en este frame se debe ejecutar detección."""
        self._frame_counter += 1
        return (self._frame_counter % self.cada_n_frames) == 0

    def actualizar_fps(self) -> float:
        """Calcula y devuelve el FPS suavizado."""
        ahora = time.perf_counter()
        delta = ahora - self._tiempo_anterior
        self._tiempo_anterior = ahora

        if delta > 0:
            fps_inst = 1.0 / delta
            # Filtro exponencial para suavizar saltos de FPS.
            self._fps_suavizado = (
                self._alpha * fps_inst + (1.0 - self._alpha) * self._fps_suavizado
            )
        return self._fps_suavizado

    @property
    def fps(self) -> float:
        return self._fps_suavizado

    def reiniciar(self) -> None:
        self._frame_counter = 0
        self._tiempo_anterior = time.perf_counter()
        self._fps_suavizado = 0.0
