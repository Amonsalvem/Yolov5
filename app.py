import io
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ------------------ UI ------------------
st.set_page_config(page_title="YOLO Detector", layout="wide")
st.title("🔎 YOLO Object Detection")

source = st.radio("Fuente de imagen", ["Subir archivo", "URL"], horizontal=True)
conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)

img_pil = None
if source == "Subir archivo":
    up = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg","jpeg","png"])
    if up is not None:
        img_pil = Image.open(up).convert("RGB")
else:
    url = st.text_input("Pega la URL de una imagen:")
    if url:
        try:
            # Leer con OpenCV para tolerar algunos headers
            data = np.frombuffer(st.session_state.get("raw_bytes", b""), dtype=np.uint8)
            # Si no hay caché previa, intenta cv2 directamente
            if data.size == 0:
                import urllib.request
                with urllib.request.urlopen(url) as resp:
                    data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            st.warning("No pude descargar/decodificar la imagen de esa URL.")

# ------------------ Modelo ------------------
@st.cache_resource
def load_model():
    """
    Intento 1: Ultralytics (YOLO >= v8).
    Intento 2: YOLOv5 vía torch.hub (clásico).
    """
    try:
        from ultralytics import YOLO  # si está disponible
        m = YOLO("yolov5s.pt")  # compatible en la mayoría de entornos
        return ("ultralytics", m)
    except Exception:
        pass

    # Fallback a yolov5 clásico
    import torch
    m = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return ("yolov5", m)

backend, model = load_model()

def run_inference(pil_img, conf=0.25):
    """
    Devuelve:
      annotated_rgb (np.ndarray HxWx3, RGB),
      rows (lista de diccionarios con predicciones)
    Soporta ambas APIs.
    """
    annotated_rgb = np.array(pil_img)  # por si hay fallback
    rows = []

    if backend == "ultralytics":
        # Ultralytics (Results list-like)
        results = model.predict(pil_img, conf=conf, verbose=False)
        # 1º elemento
        r0 = results[0]
        # Imagen anotada
        try:
            annotated_rgb = r0.plot()  # np.ndarray en RGB
        except Exception:
            annotated_rgb = np.array(pil_img)

        # Extraer cajas
        try:
            boxes = r0.boxes
            names = r0.names if hasattr(r0, "names") else model.names
            if boxes is not None:
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    confv = float(b.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    rows.append({
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                        "confidence": round(confv, 3),
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1),
                    })
        except Exception:
            pass

    else:
        # YOLOv5 clásico
        results = model(pil_img, size=640)
        # Ajustar confianza si el modelo lo soporta (algunos builds no)
        try:
            model.conf = conf
        except Exception:
            pass

        # Imagen anotada (API v5)
        annotated_rgb = np.array(pil_img)
        try:
            # En v5: results.render() devuelve lista BGR y setea results.ims/imgs
            rendered = results.render()
            if isinstance(rendered, list) and len(rendered):
                annotated_bgr = rendered[0]
            else:
                # Algunas versiones escriben en .ims o .imgs
                annotated_bgr = None
                if hasattr(results, "ims") and results.ims:
                    annotated_bgr = results.ims[0]
                if annotated_bgr is None and hasattr(results, "imgs") and results.imgs:
                    annotated_bgr = results.imgs[0]
            if annotated_bgr is not None:
                annotated_rgb = annotated_bgr[..., ::-1]
        except Exception:
            pass

        # Parse de predicciones (pandas via .pandas().xyxy[0])
        try:
            df = results.pandas().xyxy[0]
            names = model.names if hasattr(model, "names") else {}
            for _, r in df.iterrows():
                cls_id = int(r["class"])
                rows.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                    "confidence": round(float(r["confidence"]), 3),
                    "x1": round(float(r["xmin"]), 1),
                    "y1": round(float(r["ymin"]), 1),
                    "x2": round(float(r["xmax"]), 1),
                    "y2": round(float(r["ymax"]), 1),
                })
        except Exception:
            pass

    # Asegurar tipo uint8
    if annotated_rgb.dtype != np.uint8:
        annotated_rgb = np.clip(annotated_rgb, 0, 255).astype(np.uint8)
    return annotated_rgb, rows

# ------------------ Ejecución ------------------
if img_pil is not None:
    col1, col2 = st.columns(2)
    annotated_rgb, rows = run_inference(img_pil, conf=conf_thres)

    with col1:
        st.subheader("Imagen con detecciones")
        # ⚠️ NO usar use_container_width (rompe en tu runtime)
        st.image(annotated_rgb)

    with col2:
        st.subheader("Objetos detectados")
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("Sin detecciones (o no se pudo parsear la salida).")
else:
    st.info("Sube una imagen o pega una URL para comenzar.")
