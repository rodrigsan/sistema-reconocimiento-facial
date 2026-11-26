# Sistema de Reconocimiento Facial en Vivo para Eventos

Proyecto académico de reconocimiento facial en tiempo real desarrollado en Python y OpenCV.
Permite identificar asistentes de un evento proyectando su nombre y profesión en una pantalla
o TV/LED, usando como fuente de video una webcam local o el stream IP/HTTP de un celular
(IP Webcam / DroidCam).

---

## 📋 Tabla de contenidos

1. [Requisitos previos](#requisitos-previos)
2. [Clonar el repositorio](#clonar-el-repositorio)
3. [Crear entorno virtual (Python 3.11)](#crear-entorno-virtual-python-311)
4. [Instalar dependencias](#instalar-dependencias)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Ejecutar el sistema principal](#ejecutar-el-sistema-principal)
7. [Registrar asistentes](#registrar-asistentes)
8. [Ejecutar pruebas unitarias](#ejecutar-pruebas-unitarias)
9. [Teclas rápidas durante la ejecución](#teclas-rápidas-durante-la-ejecución)
10. [Checklist para el evento](#checklist-para-el-evento)

---

## 1. Requisitos previos

- **Python 3.11** (recomendado para compatibilidad con DeepFace y OpenCV)
- **Git** (para clonar el repositorio)
- Una cámara web local o un celular con IP Webcam / DroidCam instalado
- (Opcional) Una pantalla externa o TV/LED para proyección en pantalla completa

---

## 2. Clonar el repositorio

Desde una terminal (PowerShell en Windows, o Terminal en Linux/macOS):

```bash
git clone https://github.com/rodrigsan/sistema-reconocimiento-facial.git

```



---

## 3. Crear entorno virtual (Python 3.11)

Es importante usar Python 3.11 para evitar incompatibilidades con algunas dependencias
(DeepFace / TensorFlow). Si tienes varias versiones de Python instaladas, especifica la 3.11:

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Verifica que el entorno está activado: debe aparecer `(.venv)` al inicio de la línea de comandos.

---

## 4. Instalar dependencias

Con el entorno virtual activado, instala todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

Esto instalará, entre otras:
- `opencv-python` (captura de video, detección facial, ventanas GUI)
- `deepface` (embeddings faciales con modelo Facenet512)
- `numpy`, `pandas` (manipulación de datos y CSV)
- `pytest` (pruebas unitarias)

> La primera ejecución de `main.py` descargará automáticamente los pesos del modelo
> `Facenet512` de DeepFace (~130 MB) a `C:\Users\<usuario>\.deepface\weights\`.
> Las ejecuciones siguientes usarán el caché local y serán más rápidas.

---

## 5. Estructura del proyecto

```
DetectFace/
├── main.py                    # Orquestador principal (ciclo de reconocimiento en vivo)
├── requirements.txt           # Dependencias de Python
├── README.md                  # Esta guía
├── .gitignore                 # Archivos y carpetas ignorados por Git
├── data/
│   ├── asistentes.csv         # Base de datos de asistentes (UTF-8: id,nombre,profesion,foto)
│   └── embeddings_cache.pkl   # Caché de embeddings (se regenera automáticamente, no subir)
├── fotos_registrados/         # Imágenes de referencia de asistentes registrados
├── src/
│   ├── __init__.py
│   ├── stream_capture.py      # abrir_stream / leer_frame (captura de video)
│   ├── face_detector.py       # detectar_rostros (OpenCV Haar cascades)
│   ├── embeddings_index.py    # cargar_indice / extraer_embedding (DeepFace Facenet512)
│   ├── matcher.py             # buscar_coincidencia (similitud coseno, umbral <= 0.38)
│   ├── overlay_renderer.py    # dibujar_overlay (recuadros verde/naranja + FPS)
│   ├── display_output.py      # configurar_ventana / mostrar_frame (salida de video)
│   ├── performance.py         # ControladorRitmo (control de FPS, cada N frames)
│   └── registrar.py           # Script auxiliar para registrar asistentes (--img / --stream)
└── tests/
    ├── __init__.py
    ├── test_asistentes_csv.py
    ├── test_matcher.py
    └── test_simulacion_video.py
```

---

## 6. Ejecutar el sistema principal

Opcion A (la que use): Con el entorno virtual activado, ejecuta:

### Con webcam local (recomendado para pruebas iniciales)

```bash
python main.py
```



Opcion B: Con stream IP/HTTP de celular (IP Webcam / DroidCam)

```powershell
# Windows (PowerShell)
$env:STREAM_URL = "http://192.168.43.1:8080/video"
python main.py
```

```bash
# Linux / macOS
export STREAM_URL="http://192.168.43.1:8080/video"
python main.py
```

> Antes de ejecutar con stream IP, verifica que la URL sea accesible abriéndola en el
> navegador de la laptop o en VLC.

### Pantalla completa para TV/LED

Por defecto `main.py` inicia en modo ventana (`FULLSCREEN = False`) para facilitar el
desarrollo. Para activar pantalla completa:

- Edita `main.py` y cambia `FULLSCREEN = False` por `FULLSCREEN = True`, **o**
- Presiona la tecla `F` durante la ejecución para alternar entre ventana y pantalla completa.

Configura la pantalla de la laptop como **Extender** (no duplicar) y arrastra la ventana
hacia la TV/LED antes de activar pantalla completa.

---

## 7. Registrar asistentes

El archivo `data/asistentes.csv` viene vacío (solo encabezados) para que registres tus
propios asistentes. Usa el script `src/registrar.py`:

Opcion A: Desde un archivo de imagen

```bash
python src/registrar.py --nombre "Juan Pérez" --profesion "Ingeniero" --img ruta/a/foto.jpg
```

O como módulo:

```bash
python -m src.registrar --nombre "Juan Pérez" --profesion "Ingeniero" --img ruta/a/foto.jpg
```




Opcion B (la que use personalmente) : Desde el stream de video (vista previa en vivo)

```bash
python src/registrar.py --nombre "Juan Pérez" --profesion "Ingeniero" --stream
```

- Aparecerá una ventana con la vista previa de la cámara.
- Presiona **ESPACIO** para capturar la foto y registrar al asistente.
- Presiona **Q** para cancelar.

La foto se guarda en `fotos_registrados/` y la fila se agrega a `data/asistentes.csv`.

---

## 8. Ejecutar pruebas unitarias

Con el entorno virtual activado:

```bash
pytest tests/
```

Se ejecutarán 14 pruebas que cubren:
- Validación del CSV de asistentes (columnas, codificación UTF-8, filas)
- Funcionamiento del matcher de similitud coseno (coincidencia, sin coincidencia, índice vacío)
- Renderizado de overlays con y sin detecciones

---

## 9. Teclas rápidas durante la ejecución de `main.py`

| Tecla | Acción |
|-------|--------|
| `Q` / `ESC` | Salir del programa |
| `R` | Recargar `asistentes.csv` en caliente (recalcula embeddings si es necesario) |
| `F` | Alternar pantalla completa (FULLSCREEN) |

---

## 10. Checklist para el evento

- [ ] Conectar la laptop al Wi-Fi o hotspot del celular (no depender del Wi-Fi del recinto).
- [ ] En el celular sobre trípode, abrir IP Webcam o DroidCam y anotar la IP local.
- [ ] Probar la URL en el navegador de la laptop o en VLC antes de ejecutar `main.py`.
- [ ] Registrar asistentes con fotos frontales y bien iluminadas usando `registrar.py`.
- [ ] Configurar la pantalla de la laptop como "Extender" y arrastrar la ventana hacia la TV/LED.
- [ ] Hacer un ensayo general de 30 minutos con el hotspot/router real, celular en trípode y TV/LED por HDMI.
- [ ] Calibrar distancia de cámara, iluminación y umbral de reconocimiento (por defecto `0.38`).

---

## Notas técnicas

- **Umbral de reconocimiento:** distancia coseno <= 0.38 (menor = más parecido).
- **Modelo de embeddings:** DeepFace `Facenet512` con detector backend `opencv`.
- **Ventana:** `"Evento - Identificacion en Vivo"` (sin tilde para evitar problemas de encoding en Windows).
- **Caché:** `data/embeddings_cache.pkl` se crea automáticamente; no es necesario subirlo al repositorio.
