"""tests/test_asistentes_csv.py

Pruebas unitarias para el archivo data/asistentes.csv:
- Existencia del archivo
- Columnas requeridas: id, nombre, profesion, foto
- Codificación UTF-8
"""

from __future__ import annotations

import csv
import os
import sys

# Permite ejecutar pytest desde la raíz o como python -m tests.test_asistentes_csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = os.path.join("data", "asistentes.csv")
COLUMNAS_REQUERIDAS = {"id", "nombre", "profesion", "foto"}


def test_csv_existe():
    assert os.path.exists(CSV_PATH), f"El archivo {CSV_PATH} no existe"


def test_csv_columnas():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        assert lector.fieldnames is not None
        columnas = set(lector.fieldnames)
        assert COLUMNAS_REQUERIDAS.issubset(columnas), (
            f"Faltan columnas. Requeridas: {COLUMNAS_REQUERIDAS}, "
            f"Encontradas: {columnas}"
        )


def test_csv_utf8_valido():
    """Verifica que el CSV se pueda leer como UTF-8 sin errores de decodificación."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        contenido = f.read()
    assert len(contenido) > 0
    assert "id,nombre,profesion,foto" in contenido


def test_csv_tiene_filas():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        filas = list(lector)
    assert len(filas) > 0, "El CSV debe tener al menos una fila de datos"


def test_csv_filas_tienen_datos():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            assert fila["id"].strip(), "El id no puede estar vacío"
            assert fila["nombre"].strip(), "El nombre no puede estar vacío"
            assert fila["profesion"].strip(), "La profesión no puede estar vacía"
            assert fila["foto"].strip(), "La foto no puede estar vacía"
