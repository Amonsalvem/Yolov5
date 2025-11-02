# app.py — YOLOv5 (CPU) en Streamlit, con parche de imagen para Streamlit Cloud
# Compatible: Python 3.10, torch==1.12.1+cpu, yolov5==7.0.9

# ------------------------- Imports -------------------------
import os
import sys
import io
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Parche de imagen (evita TypeError) ----------------
# Todas las llamadas a st.image pasan por este wrapper seguro.
_ST_IMAGE = st.image  # guardamos referencia

def _to_pil_rgb(x):
    """Convierte PIL/NumPy (RGB/BGR/GRAY/RGBA) a PIL RGB uint8 de forma robusta."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # 2D -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]

        # Intentar BGR->RGB (si viene de OpenCV)
        try:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        return Image.fromarray(arr)

    raise TypeError("st.image data must be PIL.Image or np.ndarray (H,W,[3|4])")

def _st_image_safe(data, *args, **kwargs):
    """Renderiza por bytes PNG para evitar el error de métricas en Streamlit Cloud."""
    pil = _to_pil_rgb(data)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    kwargs.pop("channels", None)                # no usar 'channels'
    kwargs.setdefault("use_container_width", True)
    return _ST_IMAGE(buf.getvalue(), **kwargs)

# Monkey-patch
st.image = _st_image_safe
# ---------------- Fin parche de imagen ---------------------

# ------------------------- Config UI -----------------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real (YOLOv5)",
    page_icon="🧿",
    layout="wide",
)

st.title("🧿 Detección de Objetos en Imágenes (YOLOv5)")

with st.sidebar:
    st.header("Parámetros de detección")
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    class_agnostic = st.checkbox("NMS class-agnostic", value=False)
    multi_label = st.checkbox("Múltiples etiquetas por caja", value=False)
    max_det = st.number_input("Detecciones máximas", 1, 3000, 1000, 1)

# --------------------- Carga del modelo --------------------
@st.cache_resource(show_spinner=True)
def load_model():
    """
    Carga yolov5s desde archivo local (yolov5s.pt) usando el paquete 'yolov5==7.0.9'.
    """
    import yolov5  # provisto por la dependencia 'yolov5==7.0.9'
    model = yolov5.load("yolov5s.pt")  # el archivo debe existir en el repo
    model.to("cpu")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

# Seteo de hiperparámetros dinámicos
model.conf = float(conf)       # confianza
model.iou = float(iou)         # NMS IoU
model.classes = None           # detectar todas
model.max_det = int(max_det)
model.agnostic = bool(class_agnostic)
model.multi_label = bool(multi_label)

# ------------------------- Entrada -------------------------
st.subheader("📷 Captura o sube una imagen")
col1, col2 = st.columns([1, 1])

with col1:
    cam_file = st.camera_input("Usa tu cámara (opcional)")

with col2:
    up_file = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

# Determinar imagen fuente
raw_img = None
if cam_file is not None:
    raw_img = Image.open(cam_file).convert("RGB")
elif up_file is not None:
    raw_img = Image.open(up_file).convert("RGB")

if raw_img is None:
    st.info("Toma una foto o sube una imagen para ejecutar la detección.")
    st.stop()

# Mostrar la imagen fuente (usa el parche automáticamente)
st.caption("Imagen de entrada")
st.image(raw_img)

# ----------------------- Inferencia ------------------------
with st.spinner("Detectando objetos..."):
    # El modelo acepta PIL directamente
    results = model(raw_img, size=640)

# ---------------------- Resultados -------------------------
# Render del resultado (results.render() retorna listas de arrays BGR)
rendered = results.render()[0]  # numpy BGR
st.subheader("🖼️ Imagen con detecciones")
st.image(rendered)  # parche maneja BGR -> RGB

# Tabla con predicciones
st.subheader("📄 Detecciones (formato Pandas)")
try:
    df = results.pandas().xyxy[0]
    # Renombrar columnas a español amistoso
    rename = {
        "xmin": "xmin",
        "ymin": "ymin",
        "xmax": "xmax",
        "ymax": "ymax",
        "confidence": "confianza",
        "class": "clase_id",
        "name": "clase",
    }
    df = df.rename(columns=rename)
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.warning(f"No fue posible construir la tabla de detecciones: {e}")

st.success("Listo ✅")
