# app.py
# ------------------------------------------------------------------
# Detección de objetos con YOLOv5 (CPU) en Streamlit
# Compatible con: Python 3.10, torch==1.12.1, yolov5==7.0.9
# ------------------------------------------------------------------

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st

# ------------ Configuración de la página ----------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown(
    "Esta aplicación usa **YOLOv5** para detectar objetos en una imagen capturada con tu cámara. "
    "Ajusta los parámetros en la barra lateral y toma una foto para ver los resultados."
)

# ------------ Carga del modelo (cacheado) ----------
@st.cache_resource
def load_yolov5_model(model_path: str = "yolov5s.pt"):
    # 1) Si tienes el peso local, lo usa; si no, descarga desde torch.hub
    try:
        import yolov5
        # yolov5.load devuelve un wrapper que acepta ndarrays de OpenCV
        model = yolov5.load(model_path) if os.path.exists(model_path) else yolov5.load("yolov5s")
    except Exception:
        # Fallback a torch.hub (requiere internet en primer arranque)
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# ------------ Sidebar de parámetros ----------
st.sidebar.title("Parámetros de detección")
conf = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider("Umbral IoU",      0.0, 1.0, 0.45, 0.01)
agnostic = st.sidebar.checkbox("NMS class-agnostic", False)
multi_label = st.sidebar.checkbox("Múltiples etiquetas por caja", False)
max_det = st.sidebar.number_input("Detecciones máximas", 10, 2000, 1000, 10)

# Aplica parámetros al modelo (si existen en la versión instalada)
for attr, val in [
    ("conf", conf),
    ("iou", iou),
    ("agnostic", agnostic),
    ("multi_label", multi_label),
    ("max_det", max_det),
]:
    if hasattr(model, attr):
        setattr(model, attr, val)

# ------------ UI principal  ----------
main = st.container()
with main:
    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        # Decodificar bytes a imagen OpenCV (BGR)
        img_bytes = picture.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            results = model(img_bgr)

        # ---------- Renderizado seguro ----------
        # YOLOv5 (repo original) modifica results.imgs in-place y devuelve None.
        # Por claridad, tomamos la imagen anotada desde results.render().
        try:
            rendered_list = results.render()  # list[np.ndarray BGR]
            annotated_bgr = rendered_list[0] if isinstance(rendered_list, list) else results.imgs[0]
        except Exception:
            # Fallback: si por alguna razón no existiera render(), usa la original
            annotated_bgr = img_bgr

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen con detecciones")
            st.image(annotated_rgb, channels="RGB", use_container_width=True)

        # ---------- Tabla/conteo de detecciones ----------
        with col2:
            st.subheader("Objetos detectados")

            # Compatibilidad de API:
            # - yolov5<=7 usa results.pred[0] (tensor Nx6 [x1,y1,x2,y2,conf,cls])
            # - También existe results.xyxy[0] con las mismas columnas
            preds = None
            if hasattr(results, "pred"):
                preds = results.pred[0]
            elif hasattr(results, "xyxy"):
                preds = results.xyxy[0]

            if preds is None or len(preds) == 0:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            else:
                # Tensores -> numpy
                if hasattr(preds, "cpu"):
                    preds = preds.cpu().numpy()

                boxes = preds[:, :4]
                scores = preds[:, 4]
                classes = preds[:, 5].astype(int)

                # Nombres de clases
                names = getattr(model, "names", {})

                # Conteo por clase + confianza media
                df = (
                    pd.DataFrame({
                        "cls": classes,
                        "score": scores
                    })
                    .assign(Categoría=lambda d: d["cls"].map(lambda i: names.get(i, str(i))))
                    .groupby(["cls", "Categoría"], as_index=False)
                    .agg(Cantidad=("score", "count"), **{"Confianza promedio": ("score", "mean")})
                    .drop(columns=["cls"])
                )
                # Formatea confianza
                df["Confianza promedio"] = df["Confianza promedio"].map(lambda x: f"{x:.2f}")

                st.dataframe(df, use_container_width=True)

                # Gráfico simple de barras (Streamlit)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección con YOLOv5 en CPU. "
           "Desarrollada con Streamlit y PyTorch.")
