# app.py
# ------------------------------------------------------------------------------
# Detección de objetos con YOLOv5 (CPU) en Streamlit
# Compatible: Python 3.10, torch==1.12.1, torchvision==0.13.1, yolov5==7.0.9
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Configuración de la página ----------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown(
    "Esta aplicación usa **YOLOv5** para detectar objetos en una imagen capturada con la cámara. "
    "Ajusta los parámetros en la barra lateral y toma una foto para ver los resultados."
)

# ---------------- Utilidades ----------------
def to_rgb_ndarray(img_bgr) -> np.ndarray:
    """Garantiza un ndarray RGB uint8 (H,W,3) listo para st.image(..., channels='RGB')."""
    if img_bgr is None:
        return None
    arr = np.asarray(img_bgr)
    # Si viene en lista, quedarnos con el primero
    if isinstance(img_bgr, list):
        arr = np.asarray(img_bgr[0])
    # Si viene como tensor, pasarlo a numpy
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    # Si es 2D (grises), convertir a BGR
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    # Si es RGB por error, lo convertimos a BGR y luego a RGB para normalizar
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Detectar heurísticamente si ya es RGB; igualaremos a RGB al final
        pass
    # Asegurar tipo uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Convertir de BGR->RGB (la mayoría de flujos de OpenCV están en BGR)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return rgb

# ---------------- Carga de modelo (cacheado) ----------------
@st.cache_resource
def load_yolov5(model_path="yolov5s.pt"):
    """Carga YOLOv5: intenta paquete 'yolov5'; si falla, usa torch.hub."""
    try:
        import yolov5
        try:
            model = yolov5.load(model_path)  # usa el .pt local si existe
        except Exception:
            model = yolov5.load("yolov5s")   # pesos por defecto
        return model
    except Exception:
        device = torch.device("cpu")
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)
        return model

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5("yolov5s.pt")

if model is None:
    st.error("No se pudo cargar el modelo. Verifica dependencias y vuelve a intentar.")
    st.stop()

# ---------------- Sidebar / Parámetros ----------------
st.sidebar.title("Parámetros de detección")
conf_slider = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou_slider  = st.sidebar.slider("Umbral IoU",       0.0, 1.0, 0.45, 0.01)

try:
    model.conf = conf_slider
    model.iou  = iou_slider
    model.agnostic    = st.sidebar.checkbox("NMS class-agnostic", False)
    model.multi_label = st.sidebar.checkbox("Múltiples etiquetas por caja", False)
    model.max_det     = st.sidebar.number_input("Detecciones máximas", 10, 2000, 1000, 10)
except Exception:
    st.sidebar.caption("Algunas opciones avanzadas no están disponibles en esta build.")

# ---------------- Captura de imagen ----------------
picture = st.camera_input("Capturar imagen", key="camera")

if not picture:
    st.info("Toma una foto con la cámara para iniciar la detección.")
    st.stop()

# Decodificar a BGR (OpenCV)
bytes_data = picture.getvalue()
img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("No se pudo decodificar la imagen.")
    st.stop()

# ---------------- Inferencia ----------------
with st.spinner("Detectando objetos..."):
    results = model(img_bgr)

# ---------------- Obtener imagen anotada de forma robusta ----------------
annotated_bgr = None

# 1) results.render() suele devolver lista de BGR
try:
    rendered = results.render()
    if isinstance(rendered, list) and len(rendered) > 0:
        annotated_bgr = rendered[0]
except Exception:
    pass

# 2) Fallback: algunas versiones exponen results.imgs
if annotated_bgr is None and hasattr(results, "imgs"):
    imgs = results.imgs
    if isinstance(imgs, list) and len(imgs) > 0:
        annotated_bgr = imgs[0]

# 3) Último recurso: mostrar la foto original
if annotated_bgr is None:
    annotated_bgr = img_bgr

# Convertir a RGB ndarray uint8 sí o sí
annotated_rgb = to_rgb_ndarray(annotated_bgr)

# ---------------- Mostrar resultados ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen con detecciones")
    # 🔧 Aquí forzamos el formato que Streamlit espera:
    st.image(annotated_rgb, channels="RGB", use_container_width=True)

with col2:
    st.subheader("Objetos detectados")

    # Compatibilidad de predicciones
    pred = None
    if hasattr(results, "pred"):
        pred = results.pred[0]
    elif hasattr(results, "xyxy"):
        pred = results.xyxy[0]

    # Nombres de clases
    label_names = None
    if hasattr(results, "names"):
        label_names = results.names
    elif hasattr(model, "names"):
        label_names = model.names

    rows = []
    if pred is not None and len(pred) > 0 and label_names is not None:
        try:
            cls_col  = pred[:, 5].cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)[:, 5]
            conf_col = pred[:, 4].cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)[:, 4]
        except Exception:
            arr = np.asarray(pred)
            cls_col, conf_col = arr[:, 5], arr[:, 4]

        unique_cls, counts = np.unique(cls_col.astype(int), return_counts=True)
        for c, n in zip(unique_cls, counts):
            name = label_names.get(int(c), str(int(c))) if isinstance(label_names, dict) else label_names[int(c)]
            avg_conf = float(conf_col[cls_col == c].mean()) if np.any(cls_col == c) else 0.0
            rows.append({"Categoría": name, "Cantidad": int(n), "Confianza promedio": f"{avg_conf:.2f}"})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Categoría")["Cantidad"])
    else:
        st.info("No se detectaron objetos con los parámetros actuales. "
                "Prueba a bajar el umbral de confianza.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Desarrollado con Streamlit y YOLOv5 (CPU).")
