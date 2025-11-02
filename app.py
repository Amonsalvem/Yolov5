# app.py — YOLOv5 + Streamlit (CPU) con parche forzado de st.image
# Probado con: Python 3.10, torch==1.12.1+cpu, torchvision==0.13.1, yolov5==7.0.9

# ------------------------- IMPORTS BÁSICOS -------------------------
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ===================== PARCHE FORZADO DE st.image =====================
# Este parche se aplica ANTES de cualquier uso de st.image.
# Convierte PIL/NumPy a PNG en memoria y elimina 'channels' para evitar el error.
_ORIG_ST_IMAGE = st.image

def _safe_to_pil_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):  # ya es binario/PNG/JPG
        buf = io.BytesIO(x)
        try:
            im = Image.open(buf)
            return im.convert("RGB")
        except Exception:
            # Si no es imagen válida, se renderiza como está
            return None
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
    return None

def _st_image_safe(data, *args, **kwargs):
    try:
        pil = _safe_to_pil_rgb(data)
        if pil is None:
            kwargs.pop("channels", None)
            kwargs.setdefault("use_container_width", True)
            return _ORIG_ST_IMAGE(data, **kwargs)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        kwargs.pop("channels", None)
        kwargs.setdefault("use_container_width", True)
        return _ORIG_ST_IMAGE(buf.getvalue(), **kwargs)
    except Exception as e:
        return _ORIG_ST_IMAGE(f"[Render error] {e}", **{"use_container_width": True})

# Monkey-patch global
st.image = _st_image_safe
# =================== FIN PARCHE FORZADO DE st.image ===================

# ------------------------- RESTO DE IMPORTS --------------------------
import torch
import pandas as pd

# ------------------------- CONFIG DE PÁGINA --------------------------
st.set_page_config(page_title="YOLOv5 Streamlit (CPU)", page_icon="🧿", layout="wide")
st.title("🧿 Detección de Objetos (YOLOv5)")

with st.sidebar:
    st.header("Parámetros de detección")
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    class_agnostic = st.checkbox("NMS class-agnostic", value=False)
    multi_label = st.checkbox("Múltiples etiquetas por caja", value=False)
    max_det = st.number_input("Detecciones máximas", 1, 3000, 1000, 1)

# -------------------------- CARGA DEL MODELO -------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Requiere yolov5==7.0.9 en requirements
    import yolov5
    model = yolov5.load("yolov5s.pt")   # archivo en la raíz del repo
    model.to("cpu")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

# Seteo dinámico
model.conf = float(conf)
model.iou = float(iou)
model.agnostic = bool(class_agnostic)
model.multi_label = bool(multi_label)
model.max_det = int(max_det)
model.classes = None

# ---------------------------- ENTRADAS -------------------------------
st.subheader("📷 Entrada")
c1, c2 = st.columns(2)
with c1:
    cam = st.camera_input("Usa tu cámara (opcional)")
with c2:
    up = st.file_uploader("O sube una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

src_img = None
if cam is not None:
    src_img = Image.open(cam).convert("RGB")
elif up is not None:
    src_img = Image.open(up).convert("RGB")

if src_img is None:
    st.info("Toma una foto o sube una imagen para ejecutar la detección.")
    st.stop()

st.caption("Imagen de entrada")
st.image(src_img)  # <- pasa por el parche

# --------------------------- INFERENCIA ------------------------------
with st.spinner("Detectando…"):
    results = model(src_img, size=640)

# --------------------------- SALIDAS --------------------------------
# Render visual (results.render() -> lista de arrays BGR)
rendered = results.render()[0]
st.subheader("🖼️ Imagen con detecciones")
st.image(rendered)  # <- pasa por el parche

# Tabla
st.subheader("📄 Detecciones")
try:
    df = results.pandas().xyxy[0].rename(
        columns={
            "xmin": "xmin",
            "ymin": "ymin",
            "xmax": "xmax",
            "ymax": "ymax",
            "confidence": "confianza",
            "class": "clase_id",
            "name": "clase",
        }
    )
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudo construir la tabla: {e}")

st.success("Listo ✅")
