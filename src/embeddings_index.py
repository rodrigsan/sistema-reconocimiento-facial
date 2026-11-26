"""src/embeddings_index.py

Índice de embeddings faciales con caché persistente.

Contratos:
    cargar_indice(csv_path: str, fotos_dir: str, cache_path: str) -> list[dict]
    extraer_embedding(rostro_bgr: np.ndarray) -> np.ndarray | None

Modelo: DeepFace Facenet512, detector backend: opencv.
Similitud: coseno (umbral <= 0.38).
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from deepface import DeepFace


def extraer_embedding(rostro_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Calcula el vector facial de un recorte BGR usando DeepFace Facenet512.

    Args:
        rostro_bgr: Recorte BGR del rostro.

    Returns:
        Vector de embedding (numpy array) o None si no se pudo calcular.
    """
    if rostro_bgr is None or rostro_bgr.size == 0:
        return None

    try:
        # DeepFace.represent espera RGB o BGR según implementación;
        # pasamos BGR directamente y dejamos que DeepFace gestione la conversión.
        resultado = DeepFace.represent(
            img_path=rostro_bgr,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )

        if not resultado or len(resultado) == 0:
            return None

        embedding = np.array(resultado[0]["embedding"], dtype=np.float32)
        # Normalizar L2 para que la similitud coseno sea equivalente al producto punto.
        norma = np.linalg.norm(embedding)
        if norma > 0:
            embedding = embedding / norma
        return embedding

    except Exception:
        # En producción se podría loguear; aquí devolvemos None para no romper el flujo.
        return None


def _cargar_cache(cache_path: str) -> dict:
    """Carga el caché de embeddings desde un archivo pickle."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _guardar_cache(cache: dict, cache_path: str) -> None:
    """Guarda el caché de embeddings en un archivo pickle."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def cargar_indice(
    csv_path: str,
    fotos_dir: str,
    cache_path: str,
) -> list[dict]:
    """Carga o actualiza el índice de embeddings desde el CSV y las fotos.

    Lee `data/asistentes.csv` (UTF-8, columnas: id,nombre,profesion,foto),
    verifica cada foto contra el caché y recalcula embeddings faltantes.

    Args:
        csv_path: Ruta al archivo CSV de asistentes.
        fotos_dir: Directorio con las fotos de referencia.
        cache_path: Ruta al archivo pickle de caché.

    Returns:
        Lista de dicts con claves: id, nombre, profesion, foto, embedding.
    """
    # Asegurar directorios
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    os.makedirs(fotos_dir, exist_ok=True)

    # Leer CSV (UTF-8)
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path, encoding="utf-8")
    columnas_requeridas = {"id", "nombre", "profesion", "foto"}
    if not columnas_requeridas.issubset(set(df.columns)):
        raise ValueError(
            f"El CSV debe tener las columnas: {columnas_requeridas}. "
            f"Encontradas: {set(df.columns)}"
        )

    cache = _cargar_cache(cache_path)
    indice: list[dict] = []

    for _, fila in df.iterrows():
        asistente_id = str(fila["id"])
        nombre = str(fila["nombre"])
        profesion = str(fila["profesion"])
        foto_nombre = str(fila["foto"])
        foto_path = os.path.join(fotos_dir, foto_nombre)

        embedding: Optional[np.ndarray] = None

        # Verificar caché por ruta de foto
        if foto_path in cache:
            emb_cached = cache[foto_path]
            if isinstance(emb_cached, np.ndarray):
                embedding = emb_cached

        # Si no hay caché o la foto no existe, intentar extraer
        if embedding is None and os.path.exists(foto_path):
            img = cv2.imread(foto_path)
            if img is not None:
                embedding = extraer_embedding(img)
                if embedding is not None:
                    cache[foto_path] = embedding

        # Guardar caché actualizado
        _guardar_cache(cache, cache_path)

        indice.append(
            {
                "id": asistente_id,
                "nombre": nombre,
                "profesion": profesion,
                "foto": foto_nombre,
                "embedding": embedding,  # puede ser None si falló la extracción
            }
        )

    return indice
