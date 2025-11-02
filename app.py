# app.py
# -----------------------------------------------------------------------------
# Detección de objetos con YOLOv5 (CPU) en Streamlit
# Probado con: Python 3.10, torch==1.12.1, torchvision==0.13.1, yolov5==7.0.9
# -----------------------------------------------------------------------------

import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------- Configuración de la página ----------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Detección de Objetos (YOLOv5)")
st.markdown(
    "Sube una imagen **o** usa la cámara. Ajusta los parámetros y visualiza las detecciones."
)

# ---------------- Barra lateral: parámetros ----------------
st.sidebar.header("Parámetros de detección")
conf_thres = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou_thres  = st.sidebar.slider("Umbral IoU",      0.0, 1.0, 0.45, 0.01)
agnostic   = st.sidebar.checkbox("NMS class-agnostic", value=False)
multi_lbl  = st.sidebar.checkbox("Múltiples etiquetas por caja", value=False)
max_det    = st.sidebar.number_input("Detecciones máximas", min_value=1, max_value=10000, value=1000, step=1)

# ---------------- Carga/entrada de imagen ----------------
col_in1, col_in2 = st.columns([1, 1])

with col_in1:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

with col_in2:
    cam = st.camera_input("O toma una foto")

image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.read()
elif cam is not None:
    image_bytes = cam.getvalue()

# Si no hay imagen, mostramos un placeholder
if image_bytes is None:
    st.info("Sube una imagen o toma una foto para iniciar.")
    st.stop()

# Decodificar a OpenCV (BGR)
file_bytes = np.frombuffer(image_bytes, np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("No se pudo decodificar la imagen.")
    st.stop()

# ---------------- Cargar/obtener el modelo ----------------
# Preferimos pesos locales (yolov5s.pt en el repo). Si no, caerá a yolov5==7.0.9.
# Nota: Streamlit Cloud permite red al instalar/usar yolov5 pip.
try:
    # yolov5 pip (interfaz de alto nivel)
    from yolov5 import YOLOv5  # paquete pip yolov5
    weights_path = os.path.join(os.path.dirname(__file__), "yolov5s.pt")
    model = YOLOv5(weights_path, device="cpu")
    use_pip_interface = True
except Exception:
    # Fallback a hub (requiere red la primera vez)
    import torch
    model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt", source="local")
    use_pip_interface = False

# ---------------- Inferencia ----------------
# Convertimos a RGB para la API de algunos backends, pero mantendremos BGR como base
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if use_pip_interface:
    # API del paquete yolov5 (YOLOv5 class)
    results = model.predict(
        img_rgb,
        size=640,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        agnostic_nms=agnostic,
        multi_label=multi_lbl,
        max_det=max_det,
    )
    # Estandarizamos el objeto "results" a la interfaz de Ultralytics
    # Dibujamos sobre una copia BGR para mantener coherencia con cv2
    annotated_bgr = img_bgr.copy()
    for pred in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = pred[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)} {conf:.2f}"
        cv2.putText(annotated_bgr, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

else:
    # API torch hub (Ultralytics) con objeto Results que soporta .render()
    # Ajustes por atributos
    try:
        model.conf = conf_thres
        model.iou = iou_thres
        model.agnostic = agnostic
        model.multi_label = multi_lbl
        model.max_det = max_det
    except Exception:
        pass

    results = model(img_rgb, size=640)
    # YOLOv5 dibuja directamente sobre sus imágenes internas
    results.render()
    # Algunas versiones exponen .ims, otras .imgs
    try:
        annotated_bgr = results.ims[0]  # BGR
    except AttributeError:
        annotated_bgr = results.imgs[0]  # BGR

# ---------------- Visualización SEGURA en Streamlit ----------------
# Convertimos BGR -> RGB, luego PIL, luego BytesIO → st.image(bytes)
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(annotated_rgb)

buf = io.BytesIO()
img_pil.save(buf, format="PNG")
buf.seek(0)

st.subheader("Imagen con detecciones")
st.image(buf.getvalue(), use_container_width=True)
