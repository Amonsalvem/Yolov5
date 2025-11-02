# app.py
# ---------------------------------------------------------------------
# YOLOv5 en Streamlit (CPU). Muestra resultados usando BYTES (no PIL)
# para evitar TypeError en Streamlit Cloud.
# ---------------------------------------------------------------------

import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------- Config página ----------------
st.set_page_config(page_title="Detección YOLOv5", page_icon="🎯", layout="wide")
st.title("🎯 Detección de Objetos (YOLOv5)")
st.markdown("Sube una imagen o usa la cámara, ajusta parámetros y mira las detecciones.")

# ---------------- Parámetros ----------------
st.sidebar.header("Parámetros de detección")
conf_thres = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
iou_thres  = st.sidebar.slider("Umbral IoU",      0.0, 1.0, 0.45, 0.01)
agnostic   = st.sidebar.checkbox("NMS class-agnostic", value=False)
multi_lbl  = st.sidebar.checkbox("Múltiples etiquetas por caja", value=False)
max_det    = st.sidebar.number_input("Detecciones máximas", 1, 10000, 1000, 1)

# ---------------- Entrada imagen ----------------
c1, c2 = st.columns(2)
with c1:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
with c2:
    cam = st.camera_input("…o toma una foto")

image_bytes = uploaded.read() if uploaded else (cam.getvalue() if cam else None)
if not image_bytes:
    st.info("Sube una imagen o toma una foto para iniciar.")
    st.stop()

# Decodificar a OpenCV (BGR)
file_bytes = np.frombuffer(image_bytes, np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("No se pudo decodificar la imagen.")
    st.stop()

# ---------------- Cargar modelo ----------------
# Intento 1: paquete pip yolov5 (preferido). Intento 2: torch hub local.
use_pip = False
model = None
try:
    from yolov5 import YOLOv5  # paquete pip (yolov5==7.0.9)
    weights = os.path.join(os.path.dirname(__file__), "yolov5s.pt")
    model = YOLOv5(weights, device="cpu")
    use_pip = True
except Exception:
    import torch
    weights = os.path.join(os.path.dirname(__file__), "yolov5s.pt")
    # source="local" usa los pesos locales, sin descargar nada
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, source="local")

# ---------------- Inferencia ----------------
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if use_pip:
    # API del paquete yolov5
    results = model.predict(
        img_rgb,
        size=640,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        agnostic_nms=agnostic,
        multi_label=multi_lbl,
        max_det=max_det,
    )
    annotated_bgr = img_bgr.copy()
    # Dibujar cajas a mano
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)} {conf:.2f}"
        cv2.putText(annotated_bgr, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
else:
    # API torch hub (Results.render())
    try:
        model.conf = conf_thres
        model.iou = iou_thres
        model.agnostic = agnostic
        model.multi_label = multi_lbl
        model.max_det = max_det
    except Exception:
        pass
    results = model(img_rgb, size=640)
    results.render()
    try:
        annotated_bgr = results.ims[0]  # BGR
    except AttributeError:
        annotated_bgr = results.imgs[0]  # BGR

# ---------------- Mostrar (SIEMPRE EN BYTES) ----------------
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(annotated_rgb)

buf = io.BytesIO()
# PNG evita pérdidas y soporta RGBA si apareciera
pil_img.save(buf, format="PNG")
buf.seek(0)

st.subheader("Imagen con detecciones")
# IMPORTANTE: NO pasar objetos PIL aquí; solo bytes
st.image(buf.getvalue(), use_container_width=True)
