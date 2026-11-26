"""main.py

Sistema de Reconocimiento Facial en Tiempo Real para Eventos.

Orquesta el ciclo principal conectando los módulos de src/:
    stream_capture, face_detector, embeddings_index, matcher,
    overlay_renderer, display_output, performance.

Teclas rápidas:
    Q / ESC  -> salir
    R        -> recargar CSV en caliente (recalcular embeddings si es necesario)
    F        -> alternar pantalla completa (FULLSCREEN)

Configuración al inicio del archivo:
    STREAM_URL  : fuente de video (default: variable de entorno STREAM_URL o "0")
    FULLSCREEN  : pantalla completa por defecto (False para desarrollo)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np

# Asegura que se puedan importar los módulos de src/ al ejecutar desde la raíz.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stream_capture import abrir_stream, leer_frame
from src.face_detector import detectar_rostros
from src.embeddings_index import cargar_indice, extraer_embedding
from src.matcher import buscar_coincidencia
from src.overlay_renderer import dibujar_overlay
from src.display_output import configurar_ventana, mostrar_frame, VENTANA_POR_DEFECTO
from src.performance import ControladorRitmo

# ---------------------------------------------------------------------------
# CONFIGURACIÓN (modificar aquí para cambiar el comportamiento por defecto)
# ---------------------------------------------------------------------------
STREAM_URL: str = os.environ.get("STREAM_URL", "0")
FULLSCREEN: bool = False  # False para desarrollo; pon True para proyección en TV/LED

RUTAS = {
    "csv": os.path.join("data", "asistentes.csv"),
    "fotos": "fotos_registrados",
    "cache": os.path.join("data", "embeddings_cache.pkl"),
}

UMBRAL_COSINE = 0.38  # distancia coseno; menor = más parecido


def _tecla_es_salida(tecla: int) -> bool:
    return tecla in (ord("q"), ord("Q"), 27)  # 27 = ESC


def _tecla_es_recarga(tecla: int) -> bool:
    return tecla in (ord("r"), ord("R"))


def _tecla_es_fullscreen(tecla: int) -> bool:
    return tecla in (ord("f"), ord("F"))


def main() -> None:
    # Asegurar directorios necesarios
    os.makedirs(RUTAS["fotos"], exist_ok=True)
    os.makedirs(os.path.dirname(RUTAS["cache"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(RUTAS["csv"]) or ".", exist_ok=True)

    # Abrir stream de video
    print(f"[INFO] Abriendo fuente de video: {STREAM_URL}")
    cap = abrir_stream(STREAM_URL)

    # Configurar ventana
    configurar_ventana(VENTANA_POR_DEFECTO, fullscreen=FULLSCREEN)
    fullscreen_actual = FULLSCREEN

    # Cargar índice de embeddings (con caché)
    print("[INFO] Cargando índice de asistentes y embeddings...")
    indice = cargar_indice(
        csv_path=RUTAS["csv"],
        fotos_dir=RUTAS["fotos"],
        cache_path=RUTAS["cache"],
    )
    print(f"[INFO] {len(indice)} asistente(s) cargado(s).")

    # Controlador de ritmo (procesar detección cada N frames)
    ritmo = ControladorRitmo(cada_n_frames=3)

    # Detecciones previas para dibujar entre frames de procesamiento
    detecciones_ultimas: list[dict] = []
    fps_actual = 0.0

    print("[INFO] Sistema listo. Presiona Q/ESC para salir, R para recargar, F para pantalla completa.")

    try:
        while True:
            ok, frame = leer_frame(cap)
            if not ok or frame is None:
                print("[WARN] No se pudo leer frame del stream.")
                break

            # Decidir si ejecutar detección/reconocimiento en este frame
            procesar = ritmo.debe_procesar()

            if procesar:
                # Detectar rostros
                rostros = detectar_rostros(frame)

                detecciones_frame: list[dict] = []
                for rostro in rostros:
                    # Extraer embedding del recorte
                    emb = extraer_embedding(rostro["crop"])

                    # Buscar coincidencia en el índice
                    match = buscar_coincidencia(
                        emb, indice, umbral=UMBRAL_COSINE
                    ) if emb is not None else {
                        "reconocido": False,
                        "nombre": "",
                        "profesion": "",
                        "score": float("inf"),
                    }

                    detecciones_frame.append({
                        "box": rostro["box"],
                        "reconocido": match["reconocido"],
                        "nombre": match["nombre"],
                        "profesion": match["profesion"],
                        "score": match["score"],
                        "confidence": rostro["confidence"],
                    })

                detecciones_ultimas = detecciones_frame

            # Calcular FPS
            fps_actual = ritmo.actualizar_fps()

            # Dibujar overlay sobre una copia del frame
            frame_sobre = dibujar_overlay(
                frame_bgr=frame,
                detecciones=detecciones_ultimas,
                fps=fps_actual,
            )

            # Mostrar frame y leer tecla
            tecla = mostrar_frame(VENTANA_POR_DEFECTO, frame_sobre)

            if _tecla_es_salida(tecla):
                print("[INFO] Saliendo por petición del usuario (Q/ESC).")
                break

            if _tecla_es_recarga(tecla):
                print("[INFO] Recargando CSV en caliente...")
                indice = cargar_indice(
                    csv_path=RUTAS["csv"],
                    fotos_dir=RUTAS["fotos"],
                    cache_path=RUTAS["cache"],
                )
                print(f"[INFO] Índice recargado: {len(indice)} asistente(s).")

            if _tecla_es_fullscreen(tecla):
                fullscreen_actual = not fullscreen_actual
                modo = "FULLSCREEN" if fullscreen_actual else "ventana"
                print(f"[INFO] Cambiado a modo: {modo}")
                cv2.setWindowProperty(
                    VENTANA_POR_DEFECTO,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen_actual else cv2.WINDOW_NORMAL,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Sistema finalizado.")


if __name__ == "__main__":
    main()
