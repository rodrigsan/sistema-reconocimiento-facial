"""src/matcher.py

Búsqueda de coincidencia facial por similitud coseno.

Contrato:
    buscar_coincidencia(embedding: np.ndarray, indice: list[dict],
                        umbral: float = 0.38) -> dict

Retorna:
    {"reconocido": bool, "nombre": str, "profesion": str, "score": float}
"""

from __future__ import annotations

import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la distancia coseno entre dos vectores (0 = idénticos, 2 = opuestos)."""
    if a is None or b is None:
        return float("inf")
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    if norma_a == 0 or norma_b == 0:
        return float("inf")
    cos_sim = float(np.dot(a, b) / (norma_a * norma_b))
    # Distancia coseno: 1 - similitud coseno
    return 1.0 - cos_sim


def buscar_coincidencia(
    embedding: np.ndarray,
    indice: list[dict],
    umbral: float = 0.38,
) -> dict:
    """Busca el rostro más parecido en el índice usando distancia coseno.

    Args:
        embedding: Vector del rostro detectado en el frame.
        indice: Lista de asistentes con embeddings precalculados.
        umbral: Distancia coseno máxima para considerar coincidencia (menor = más parecido).

    Returns:
        Dict con claves: reconocido, nombre, profesion, score.
    """
    resultado_no_reconocido = {
        "reconocido": False,
        "nombre": "",
        "profesion": "",
        "score": float("inf"),
    }

    if embedding is None:
        return resultado_no_reconocido

    mejor_score = float("inf")
    mejor_asistente = None

    for entrada in indice:
        emb_guardado = entrada.get("embedding")
        if emb_guardado is None:
            continue

        distancia = _cosine_distance(embedding, emb_guardado)
        if distancia < mejor_score:
            mejor_score = distancia
            mejor_asistente = entrada

    if mejor_asistente is not None and mejor_score <= umbral:
        return {
            "reconocido": True,
            "nombre": mejor_asistente["nombre"],
            "profesion": mejor_asistente["profesion"],
            "score": float(mejor_score),
        }

    return resultado_no_reconocido
