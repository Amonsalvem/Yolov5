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

# ---------------- Carga de modelo (cacheado) ----------------
@st.cache_resource
def load_yolov5(model_path="yolov5s.pt"):
    """
    Carga robusta de YOLOv5:
    1) Intenta con el paquete 'yolov5' (pypi) si está disponible.
    2) Fallback: torch.hub ultralytics/yolov5.
    """
    try:
        import yolov5
        try:
            model = yolov5.load(model_path)  # usa el .pt local si existe
        except Exception:
            # Fallback a los pesos por defecto si no encuentra el archivo
            model = yolov5.load("yolov5s")
        return model
    except Exception:
        # Fallback torch.hub
        device = torch.device("cpu")
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True
        ).to(device)
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

# Ajustes del modelo (si están disponibles)
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

# ---------------- Renderizado robusto para st.image ----------------
# Intentamos obtener la imagen ANOTADA (con cajas) en BGR
annotated_bgr = None
try:
    rendered = results.render()  # normalmente devuelve list[np.ndarray(BGR)]
    if isinstance(rendered, list) and len(rendered) > 0:
        annotated_bgr = rendered[0]
except Exception:
    pass

if annotated_bgr is None:
    # Fallback: algunas versiones dejan la imagen ya anotada en results.imgs
    if hasattr(results, "imgs") and isinstance(results.imgs, list) and len(results.imgs) > 0:
        annotated_bgr = results.imgs[0]
    else:
        annotated_bgr = img_bgr  # último recurso (sin cajas)

# Asegurar formato correcto
if isinstance(annotated_bgr, list) and len(annotated_bgr) > 0:
    annotated_bgr = annotated_bgr[0]
if hasattr(annotated_bgr, "detach"):
    annotated_bgr = annotated_bgr.detach().cpu().numpy()
annotated_bgr = np.asarray(annotated_bgr)
if annotated_bgr.ndim == 2:
    annotated_bgr = cv2.cvtColor(annotated_bgr, cv2.COLOR_GRAY2BGR)
if annotated_bgr.dtype != np.uint8:
    annotated_bgr = np.clip(annotated_bgr, 0, 255).astype(np.uint8)

# BGR -> RGB y a PIL para Streamlit
from PIL import Image
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(annotated_rgb)

# ---------------- Mostrar resultados ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen con detecciones")
    st.image(img_pil, use_container_width=True)

# Parseo de predicciones (soporta variantes de YOLOv5)
with col2:
    st.subheader("Objetos detectados")

    # Compatibilidad: results.pred (clásico) o results.xyxy
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

    data_rows = []
    if pred is not None and len(pred) > 0 and label_names is not None:
        # pred: [x1,y1,x2,y2,conf,cls]
        try:
            cls_col = pred[:, 5].cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)[:, 5]
            conf_col = pred[:, 4].cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)[:, 4]
        except Exception:
            # Fallback muy defensivo
            arr = np.asarray(pred)
            cls_col = arr[:, 5]
            conf_col = arr[:, 4]

        # Conteo por clase
        unique_cls, counts = np.unique(cls_col.astype(int), return_counts=True)
        for c, n in zip(unique_cls, counts):
            name = label_names.get(int(c), str(int(c))) if isinstance(label_names, dict) else label_names[int(c)]
            avg_conf = float(conf_col[cls_col == c].mean()) if np.any(cls_col == c) else 0.0
            data_rows.append({"Categoría": name, "Cantidad": int(n), "Confianza promedio": f"{avg_conf:.2f}"})

    if data_rows:
        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Categoría")["Cantidad"])
    else:
        st.info("No se detectaron objetos con los parámetros actuales. "
                "Prueba a bajar el umbral de confianza.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Desarrollado con Streamlit y YOLOv5 (CPU).")
