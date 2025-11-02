# app.py
# -----------------------------------------------------------------------------
# Detección de objetos con YOLOv5 (CPU) en Streamlit
# Compatible: Python 3.10, torch==1.12.x, yolov5==7.0.9
# -----------------------------------------------------------------------------

import io
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------- Configuración de la página ----------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown(
    "Esta aplicación usa **YOLOv5** para detectar objetos en una imagen capturada con la cámara. "
    "Ajusta los parámetros en la barra lateral y toma una foto para ver los resultados."
)

# ---------------- Cargar modelo ----------------
@st.cache_resource
def load_model():
    try:
        import yolov5
    except Exception as e:
        st.error(f"No se pudo importar yolov5: {e}")
        return None

    try:
        # Si existe un peso local .pt, úsalo; si no, carga el modelo pequeño por nombre
        if os.path.exists("yolov5s.pt"):
            model = yolov5.load("yolov5s.pt")
        else:
            model = yolov5.load("yolov5s")
    except Exception as e:
        st.error(f"Error cargando modelo YOLOv5: {e}")
        return None

    # valores por defecto (se podrán ajustar en la sidebar)
    model.conf = 0.25
    model.iou = 0.45
    try:
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000
    except:
        pass

    return model

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model is None:
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Parámetros de detección")
model.conf = st.sidebar.slider("Confianza mínima", 0.00, 1.00, float(model.conf), 0.01)
model.iou = st.sidebar.slider("Umbral IoU", 0.00, 1.00, float(model.iou), 0.01)

try:
    model.agnostic = st.sidebar.checkbox("NMS class-agnostic", bool(getattr(model, "agnostic", False)))
    model.multi_label = st.sidebar.checkbox("Múltiples etiquetas por caja", bool(getattr(model, "multi_label", False)))
    model.max_det = st.sidebar.number_input("Detecciones máximas", 10, 2000, int(getattr(model, "max_det", 1000)), 10)
except:
    st.sidebar.caption("Opciones avanzadas no disponibles para esta build.")

# ---------------- Captura con cámara ----------------
picture = st.camera_input("Capturar imagen", key="camera")

if picture is None:
    st.info("Toma una foto para ejecutar la detección.")
    st.stop()

# Convertir bytes -> OpenCV BGR
bytes_data = picture.getvalue()
cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

if cv2_img is None:
    st.error("No se pudo leer la imagen capturada.")
    st.stop()

# ---------------- Inferencia ----------------
with st.spinner("Detectando objetos..."):
    try:
        results = model(cv2_img)  # acepta BGR directamente
    except Exception as e:
        st.error(f"Error durante la inferencia: {e}")
        st.stop()

# ---------------- Parseo de resultados ----------------
try:
    # 1) Imagen anotada
    # En algunas versiones .render() devuelve lista; en otras modifica internamente.
    annotated_list = None
    try:
        annotated_list = results.render()  # try: devuelve [np.ndarray BGR]
    except Exception:
        pass

    if annotated_list and len(annotated_list) > 0:
        annotated_bgr = annotated_list[0]
    else:
        # fallback para builds que guardan en .imgs
        annotated_bgr = getattr(results, "imgs", [cv2_img])[0]

    # Convertir BGR -> RGB y mostrar con PIL (sin argumento channels=)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = np.ascontiguousarray(annotated_rgb).astype(np.uint8)
    img_pil = Image.fromarray(annotated_rgb, mode="RGB")

    # 2) Tabla de conteo
    preds = results.pred[0]  # [x1,y1,x2,y2,conf,cls]
    boxes = preds[:, :4] if preds.numel() > 0 else []
    scores = preds[:, 4] if preds.numel() > 0 else []
    classes = preds[:, 5].to(torch.int64) if preds.numel() > 0 else []

    labels = getattr(model, "names", {})
    counts = {}
    for c in classes:
        idx = int(c.item())
        counts[idx] = counts.get(idx, 0) + 1

    rows = []
    for idx, qty in counts.items():
        label = labels[idx] if isinstance(labels, (list, dict)) and idx in labels else str(idx)
        # media de la confianza para esa clase
        if len(scores) > 0:
            mask = classes == idx
            conf_mean = scores[mask].mean().item()
        else:
            conf_mean = 0.0
        rows.append({"Categoría": label, "Cantidad": qty, "Confianza promedio": f"{conf_mean:.2f}"})

    df = pd.DataFrame(rows)

except Exception as e:
    st.error(f"Error al procesar los resultados: {e}")
    st.stop()

# ---------------- UI de resultados ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen con detecciones")
    # Mostrar como bytes para evitar incompatibilidades de PIL/Streamlit
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf.getvalue(), use_container_width=True)


with col2:
    st.subheader("Objetos detectados")
    if len(df) > 0:
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Categoría")["Cantidad"])
    else:
        st.info("No se detectaron objetos con los parámetros actuales. "
                "Prueba a bajar el umbral de confianza.")

# ---------------- Pie ----------------
st.markdown("---")
st.caption("**Acerca de**: Demo de detección de objetos con YOLOv5 en Streamlit (CPU).")
