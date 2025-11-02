# app.py
# ------------------------------------------------------------
# Detección de objetos con YOLOv5 (CPU) en Streamlit
# Compatible con: Python 3.10, torch==1.12.1+cpu, yolov5==7.0.9
# ------------------------------------------------------------
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Configuración de la página ----------
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


# ---------- Carga del modelo (con cache) ----------
@st.cache_resource(show_spinner=True)
def load_yolov5_model(local_weights: str = "yolov5s.pt"):
    """
    Carga el modelo YOLOv5 de forma robusta:
    1) Si existe yolov5s.pt en el repo, lo carga con yolov5.load().
    2) Si no existe, usa torch.hub (ultralytics/yolov5) con 'yolov5s' (pretrained).
    Siempre en CPU para ser compatible con Streamlit Cloud.
    """
    device = torch.device("cpu")

    try:
        import yolov5  # del paquete yolov5==7.0.9

        # 1) Pesos locales si existen
        if os.path.exists(local_weights):
            try:
                model = yolov5.load(local_weights, weights_only=False, device=device)
                model.to(device)
                return model
            except TypeError:
                # weights_only no soportado -> intento básico
                model = yolov5.load(local_weights, device=device)
                model.to(device)
                return model

        # 2) Fallback: torch.hub (descarga 'yolov5s' pretrained)
        model = torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="yolov5s",
            pretrained=True,
            force_reload=False
        )
        model.to(device)
        return model

    except Exception as e:
        st.error(f"❌ No se pudo cargar YOLOv5: {e}")
        st.stop()


with st.spinner("Cargando modelo YOLOv5…"):
    model = load_yolov5_model()

# ---------- Parámetros en sidebar ----------
with st.sidebar:
    st.header("Parámetros de detección")
    conf_thres = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)

    # Intento de exponer opciones avanzadas cuando existen
    agnostic = st.checkbox("NMS class-agnostic", False)
    multi_label = st.checkbox("Múltiples etiquetas por caja", False)
    max_det = st.number_input("Detecciones máximas", min_value=10, max_value=2000, value=1000, step=10)

# Aplicar parámetros si el objeto modelo los soporta
try:
    model.conf = conf_thres
except Exception:
    pass
try:
    model.iou = iou_thres
except Exception:
    pass
for attr, val in [("agnostic", agnostic), ("multi_label", multi_label), ("max_det", max_det)]:
    if hasattr(model, attr):
        try:
            setattr(model, attr, val)
        except Exception:
            pass

st.caption(f"**Confianza:** {conf_thres:.2f} · **IoU:** {iou_thres:.2f}")

st.markdown("---")

# ---------- Captura con la cámara ----------
picture = st.camera_input("Capturar imagen", key="camera")

if picture is None:
    st.info("Usa el botón de arriba para tomar una foto con la cámara.")
    st.stop()

# Convertir el buffer a imagen OpenCV (BGR)
bytes_data = picture.getvalue()
cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# ---------- Inferencia ----------
with st.spinner("Detectando objetos…"):
    try:
        # Llamar al modelo directamente con imagen en BGR
        results = model(cv2_img)
    except Exception as e:
        st.error(f"Error durante la detección: {e}")
        st.stop()

# ---------- Visualización ----------
col1, col2 = st.columns(2)

# Columna 1: imagen anotada
with col1:
    st.subheader("Imagen con detecciones")

    # YOLOv5 anota internamente al llamar a render()
    try:
        results.render()
        # results.imgs[0] queda en BGR (si se renderiza con cv2)
        annotated_bgr = results.imgs[0]
        # Convertir a RGB para Streamlit si es necesario
        annotated_rgb = annotated_bgr[:, :, ::-1]
        st.image(annotated_rgb, use_container_width=True)
    except Exception:
        # Fallback: mostrar la imagen original
        st.image(cv2_img[:, :, ::-1], use_container_width=True)

# Columna 2: tabla y conteo por clase
with col2:
    st.subheader("Objetos detectados")

    try:
        # Formato estable en yolov5==7.0.9
        boxes_tensor = results.xyxy[0]  # [x1,y1,x2,y2,conf,cls]
        if boxes_tensor is None or len(boxes_tensor) == 0:
            st.info("No se detectaron objetos con los parámetros actuales.")
        else:
            scores = boxes_tensor[:, 4]
            classes = boxes_tensor[:, 5].int()

            # Nombres de clases
            label_names = getattr(model, "names", None)
            if label_names is None:
                label_names = getattr(results, "names", {})

            # Agrupar por clase
            rows = []
            for cls_id in classes.unique().tolist():
                mask = (classes == cls_id)
                count = int(mask.sum().item())
                mean_conf = float(scores[mask].mean().item())
                name = label_names[int(cls_id)] if isinstance(label_names, (list, tuple, dict)) else str(int(cls_id))
                rows.append({
                    "Categoría": name,
                    "Cantidad": count,
                    "Confianza promedio": f"{mean_conf:.2f}"
                })

            df = pd.DataFrame(rows).sort_values("Cantidad", ascending=False)
            st.dataframe(df, use_container_width=True)

            # Gráfico simple
            try:
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            except Exception:
                pass

    except Exception as e:
        st.error(f"Error al procesar resultados: {e}")
        st.stop()

st.markdown("---")
st.caption(
    "App de demostración en **Streamlit** con **YOLOv5 (CPU)**. "
    "Recomendado para Python 3.10 y las versiones de librerías fijadas en `requirements.txt`."
)
