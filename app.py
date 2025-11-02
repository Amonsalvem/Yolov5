# app.py — YOLO + Streamlit compatible con APIs antiguas y nuevas

import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ---------------- Utilidad robusta para mostrar imágenes ----------------
def show_image_robusto(img):
    """
    Muestra 'img' (PIL o np.ndarray) en Streamlit tolerando:
    - versiones antiguas (sin use_container_width)
    - ausencia de 'channels'
    """
    # normalizamos a PIL RGB
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # si parece BGR -> a RGB
        try:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        pil = Image.fromarray(arr)
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        # si llega bytes/jpg/png
        try:
            pil = Image.open(io.BytesIO(img)).convert("RGB")
        except Exception:
            st.image(img)  # último recurso
            return

    # intentamos con argumentos nuevos → viejos → sin args
    try:
        st.image(pil, use_container_width=True)
    except TypeError:
        try:
            st.image(pil, use_column_width=True)
        except TypeError:
            st.image(pil)

# --------------- App UI ---------------
st.set_page_config(page_title="YOLO Streamlit", layout="wide")
st.title("🧿 Detección de objetos")

with st.sidebar:
    st.header("Parámetros de detección")
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    class_agnostic = st.checkbox("NMS class-agnostic", value=False)
    multi_label = st.checkbox("Múltiples etiquetas por caja", value=False)
    max_det = st.number_input("Detecciones máximas", 1, 3000, 1000, 1)

# --------------- Carga del modelo (soporta yolov5 o ultralytics) ---------------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Paquete yolov5 clásico
        import yolov5
        m = yolov5.load("yolov5s.pt")  # asegúrate de tener el .pt en el repo
        m.to("cpu")
        return ("yolov5", m)
    except Exception:
        # Fallback a Torch Hub
        import torch
        m = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        m.to("cpu")
        return ("yolov5_hub", m)

flavor, model = load_model()

# Ajustes (los nombres existen en yolov5; si no, ignoramos)
for attr, val in dict(conf=conf, iou=iou, agnostic=class_agnostic,
                      multi_label=multi_label, max_det=int(max_det)).items():
    try:
        setattr(model, attr, val)
    except Exception:
        pass

# --------------- Entrada ---------------
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
show_image_robusto(src_img)

# --------------- Inferencia ---------------
with st.spinner("Detectando…"):
    results = model(src_img, size=640) if flavor.startswith("yolov5") else model(src_img)

# --------------- Render de imagen anotada (manejo multi-APIs) ---------------
def obtener_anotada(results):
    """
    Devuelve np.ndarray/PIL de la imagen anotada usando la API disponible.
    Soporta:
      - yolov5 clásico: results.render() -> lista de np.ndarray (BGR)
      - ultralytics v8 (por si el paquete cambia): results[0].plot()
    """
    # YOLOv5 clásico
    try:
        if hasattr(results, "render"):
            ims = results.render()
            if isinstance(ims, list) and len(ims) > 0:
                return ims[0]  # BGR
    except Exception:
        pass

    # Ultralytics (v8)
    try:
        item0 = results[0]
        if hasattr(item0, "plot"):
            return item0.plot()  # RGB np.ndarray
    except Exception:
        pass

    return None

annot = obtener_anotada(results)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Imagen con detecciones")
    if annot is not None:
        show_image_robusto(annot)
    else:
        # Sin anotada disponible, mostramos la original
        show_image_robusto(src_img)

# --------------- Tabla de detecciones (multi-APIs) ---------------
with col2:
    st.subheader("📄 Detecciones")
    df = None
    # YOLOv5 clásico
    try:
        df = results.pandas().xyxy[0]
        df = df.rename(columns={
            "xmin": "xmin", "ymin": "ymin", "xmax": "xmax", "ymax": "ymax",
            "confidence": "confianza", "class": "clase_id", "name": "clase",
        })
    except Exception:
        # Ultralytics v8
        try:
            r0 = results[0]
            boxes = r0.boxes  # Boxes object
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            names = getattr(getattr(r0, "names", {}), "items", lambda: {})()
            # Crear DataFrame
            data = []
            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = xyxy[i].tolist()
                conf = float(confs[i])
                cid = int(clss[i])
                cname = r0.names[cid] if hasattr(r0, "names") and cid in r0.names else str(cid)
                data.append([xmin, ymin, xmax, ymax, conf, cid, cname])
            df = pd.DataFrame(data, columns=["xmin","ymin","xmax","ymax","confianza","clase_id","clase"])
        except Exception:
            df = None

    if df is None or df.empty:
        st.info("No se pudieron construir las detecciones (o no se detectaron objetos).")
    else:
        st.dataframe(df, use_container_width=True if "use_container_width" in st.image.__code__.co_varnames else True)

st.success("Listo ✅")
