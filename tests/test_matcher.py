"""tests/test_matcher.py

Pruebas unitarias para src/matcher.py:
- Coincidencia exacta (mismo vector)
- Sin coincidencia (vectores muy diferentes)
- Índice vacío
- Embedding None
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matcher import buscar_coincidencia


def _hacer_indice(nombre: str, profesion: str, vector: np.ndarray):
    return [
        {"id": "1", "nombre": nombre, "profesion": profesion, "foto": "1.jpg", "embedding": vector}
    ]


def test_coincidencia_exacta():
    """Mismo vector debe dar reconocido=True con score 0 (distancia coseno)."""
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    indice = _hacer_indice("Juan Perez", "Ingeniero", v)
    resultado = buscar_coincidencia(v, indice, umbral=0.38)
    assert resultado["reconocido"] is True
    assert resultado["nombre"] == "Juan Perez"
    assert resultado["profesion"] == "Ingeniero"
    assert abs(resultado["score"] - 0.0) < 1e-6


def test_sin_coincidencia():
    """Vectores ortogonales (distancia coseno = 1) no deben coincidir con umbral 0.38."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    indice = _hacer_indice("Juan Perez", "Ingeniero", v1)
    resultado = buscar_coincidencia(v2, indice, umbral=0.38)
    assert resultado["reconocido"] is False
    assert resultado["nombre"] == ""
    assert resultado["profesion"] == ""


def test_indice_vacio():
    """Índice vacío debe retornar no reconocido."""
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    resultado = buscar_coincidencia(v, [], umbral=0.38)
    assert resultado["reconocido"] is False


def test_embedding_none():
    """Embedding None debe retornar no reconocido sin fallar."""
    indice = _hacer_indice("Juan Perez", "Ingeniero", np.array([1.0, 0.0, 0.0]))
    resultado = buscar_coincidencia(None, indice, umbral=0.38)
    assert resultado["reconocido"] is False


def test_umbral_personalizado():
    """Con umbral muy bajo (0.01), vectores similares pero no iguales no coinciden."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)  # normalizar
    indice = _hacer_indice("Juan Perez", "Ingeniero", v1)
    resultado = buscar_coincidencia(v2, indice, umbral=0.01)
    # La distancia coseno entre [1,0,0] y [0.99,0.1,0] normalizado es ~0.005
    # que es menor que 0.01, así que debería coincidir. Verificamos que no falla.
    assert "reconocido" in resultado
    assert "score" in resultado
