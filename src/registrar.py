"""src/registrar.py

Script auxiliar para dar de alta asistentes.

Modos de uso:
    python registrar.py --img <ruta_foto> --nombre <nombre> --profesion <profesion>
    python registrar.py --stream --nombre <nombre> --profesion <profesion>
    python -m src.registrar --img ...   (desde la raíz del proyecto)

Contrato:
    registrar_asistente(nombre: str, profesion: str, ruta_foto: str,
                        csv_path: str, fotos_dir: str) -> dict
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from typing import Optional

import cv2

# Permite ejecutar como `python -m src.registrar` desde la raíz.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.display_output import mostrar_frame, configurar_ventana, VENTANA_POR_DEFECTO
from src.stream_capture import abrir_stream, leer_frame


CSV_DEFECTO = os.path.join("data", "asistentes.csv")
FOTOS_DEFECTO = "fotos_registrados"


def registrar_asistente(
    nombre: str,
    profesion: str,
    ruta_foto: str,
    csv_path: str = CSV_DEFECTO,
    fotos_dir: str = FOTOS_DEFECTO,
) -> dict:
    """Valida/copia la foto y agrega la fila a asistentes.csv.

    Args:
        nombre: Nombre del asistente.
        profesion: Profesión del asistente.
        ruta_foto: Ruta al archivo de imagen de referencia.
        csv_path: Ruta al CSV de asistentes.
        fotos_dir: Directorio donde se guardan las fotos.

    Returns:
        Dict con el resultado de la operación.
    """
    if not nombre or not nombre.strip():
        return {"ok": False, "error": "El nombre es obligatorio."}
    if not profesion or not profesion.strip():
        return {"ok": False, "error": "La profesión es obligatoria."}
    if not os.path.exists(ruta_foto):
        return {"ok": False, "error": f"No se encontró la foto: {ruta_foto}"}

    os.makedirs(fotos_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Determinar el siguiente ID
    existing_ids = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                try:
                    existing_ids.append(int(fila["id"]))
                except (ValueError, KeyError):
                    pass

    nuevo_id = max(existing_ids, default=0) + 1

    # Nombre de archivo para la foto (usa el ID para evitar colisiones)
    ext = os.path.splitext(ruta_foto)[1].lower() or ".jpg"
    nombre_archivo = f"{nuevo_id}{ext}"
    destino_foto = os.path.join(fotos_dir, nombre_archivo)

    # Copiar la foto al directorio de registros
    shutil.copy2(ruta_foto, destino_foto)

    # Agregar fila al CSV (crear con cabecera si no existe)
    escribir_cabecera = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if escribir_cabecera:
            escritor.writerow(["id", "nombre", "profesion", "foto"])
        escritor.writerow([nuevo_id, nombre.strip(), profesion.strip(), nombre_archivo])

    return {
        "ok": True,
        "id": nuevo_id,
        "nombre": nombre.strip(),
        "profesion": profesion.strip(),
        "foto": nombre_archivo,
        "ruta_foto": destino_foto,
    }


def _capturar_desde_stream(
    nombre: str,
    profesion: str,
    fuente: str,
    csv_path: str = CSV_DEFECTO,
    fotos_dir: str = FOTOS_DEFECTO,
) -> dict:
    """Muestra vista previa en vivo; ESPACIO captura, Q cancela."""
    cap = abrir_stream(fuente)
    ventana = VENTANA_POR_DEFECTO + " - Registro"
    configurar_ventana(ventana, fullscreen=False)

    capturado = None
    try:
        while True:
            ok, frame = leer_frame(cap)
            if not ok or frame is None:
                break

            # Dibujar instrucciones
            cv2.putText(
                frame,
                "ESPACIO: capturar foto   Q: cancelar",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Registrando: {nombre} | {profesion}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            mostrar_frame(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), 27):  # Q o ESC
                break
            elif tecla == 32:  # ESPACIO
                capturado = frame.copy()
                break
    finally:
        cap.release()
        cv2.destroyWindow(ventana)

    if capturado is None:
        return {"ok": False, "error": "Captura cancelada por el usuario."}

    # Guardar foto capturada
    os.makedirs(fotos_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Determinar el siguiente ID
    existing_ids = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                try:
                    existing_ids.append(int(fila["id"]))
                except (ValueError, KeyError):
                    pass

    nuevo_id = max(existing_ids, default=0) + 1
    nombre_archivo = f"{nuevo_id}.jpg"
    destino_foto = os.path.join(fotos_dir, nombre_archivo)
    cv2.imwrite(destino_foto, capturado)

    # Agregar fila al CSV
    escribir_cabecera = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if escribir_cabecera:
            escritor.writerow(["id", "nombre", "profesion", "foto"])
        escritor.writerow([nuevo_id, nombre.strip(), profesion.strip(), nombre_archivo])

    return {
        "ok": True,
        "id": nuevo_id,
        "nombre": nombre.strip(),
        "profesion": profesion.strip(),
        "foto": nombre_archivo,
        "ruta_foto": destino_foto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Registrar asistente en el sistema de reconocimiento facial."
    )
    parser.add_argument("--nombre", required=True, help="Nombre del asistente")
    parser.add_argument("--profesion", required=True, help="Profesión del asistente")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--img", help="Ruta al archivo de imagen de referencia")
    grupo.add_argument(
        "--stream",
        action="store_true",
        help="Capturar desde el stream de video (webcam o IP)",
    )
    parser.add_argument(
        "--fuente",
        default=os.environ.get("STREAM_URL", "0"),
        help="Fuente de video para --stream (default: %(default)s)",
    )
    parser.add_argument(
        "--csv", default=CSV_DEFECTO, help="Ruta al CSV de asistentes"
    )
    parser.add_argument(
        "--fotos-dir", default=FOTOS_DEFECTO, help="Directorio de fotos"
    )

    args = parser.parse_args()

    if args.stream:
        resultado = _capturar_desde_stream(
            nombre=args.nombre,
            profesion=args.profesion,
            fuente=args.fuente,
            csv_path=args.csv,
            fotos_dir=args.fotos_dir,
        )
    else:
        resultado = registrar_asistente(
            nombre=args.nombre,
            profesion=args.profesion,
            ruta_foto=args.img,
            csv_path=args.csv,
            fotos_dir=args.fotos_dir,
        )

    if resultado.get("ok"):
        print(
            f"[OK] Asistente registrado: ID={resultado['id']}, "
            f"{resultado['nombre']} ({resultado['profesion']}), foto: {resultado['foto']}"
        )
    else:
        print(f"[ERROR] {resultado.get('error', 'Error desconocido')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
