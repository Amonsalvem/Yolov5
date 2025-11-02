import os
import sys
import io
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ================================
# Configuración de página Streamlit
# ================================
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# =======================================================
# FIX de renderizado seguro para Streamlit Cloud (OBLIGADO)
# =======================================================
def _render_image_safe(img):
    """
    Acepta:
      - PIL.Image.Image
      - np.ndarray (H,W,3) en RGB o BGR
    Renderiza por bytes para evitar el TypeError en Streamlit Cloud.
    """
    # Convertir a PIL RGB
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Si viene de OpenCV es BGR → convertir a RGB
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        raise TypeError("img debe ser PIL.Image o np.ndarray HxWx3")

    # Volcar a PNG en memoria y mostrar
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf.getvalue(), use_container_width=True)

# =======================================================
# Cargar modelo YOLOv5 (compatibilidad con distintos entornos)
# =======================================================
@st.cache_resource
def load_yolov5_model(model_path: str = "yolov5s.pt"):
    """
    Intenta cargar YOLOv5 usando:
    1) paquete 'yolov5' si está instalado
    2) torch.hub (ultralytics/yolov5) como fallback
    """
    try:
        import yolov5  # si está instalado el paquete
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
        except Exception:
            # fallback a torch.hub
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
    except Exception:
        # fallback directo a torch.hub
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
            st.info(
                "Sugerencias:\n"
                "• Asegúrate de que las dependencias de PyTorch/YOLOv5 son compatibles.\n"
                "• Mantén 'ultralytics/yolov5' accesible para torch.hub.\n"
                "• Verifica que 'yolov5s.pt' exista si cargas por ruta local."
            )
            return None

# ======================
# UI: Título y descripción
# ======================
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown(
    "Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara. "
    "Ajusta los parámetros en la barra lateral para personalizar la detección."
)

# ======================
# Cargar el modelo
# ======================
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model is None:
    st.error("No se pudo cargar el modelo. Revisa dependencias y vuelve a intentarlo.")
    st.stop()

# ======================
# Sidebar parámetros
# ======================
st.sidebar.title("Parámetros")
with st.sidebar:
    st.subheader("Configuración de detección")
    # Algunos wrappers de YOLOv5 aceptan asignación directa; si no, lo hacemos vía attrs presentes
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    st.caption(f"Confianza: {conf:.2f} | IoU: {iou:.2f}")

    # Intentar setear en el modelo si expone estas propiedades
    for name, value in (("conf", conf), ("iou", iou)):
        try:
            setattr(model, name, value)
        except Exception:
            pass

    st.subheader("Opciones avanzadas")
    agnostic = st.checkbox("NMS class-agnostic", False)
    multi_label = st.checkbox("Múltiples etiquetas por caja", False)
    max_det = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    for name, value in (("agnostic", agnostic), ("multi_label", multi_label), ("max_det", int(max_det))):
        try:
            setattr(model, name, value)
        except Exception:
            pass

# ======================
# Captura y predicción
# ======================
main_container = st.container()
with main_container:
    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        # Bytes → OpenCV BGR
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            try:
                # Inference (acepta np.ndarray BGR/ RGB)
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {e}")
                st.stop()

        # ======================
        # Visualización anotada
        # ======================
        try:
            # results.render() modifica internamente la imagen y expone .imgs (lista RGB)
            results.render()
            if hasattr(results, "imgs") and len(results.imgs) > 0:
                annotated_rgb = results.imgs[0]  # np.ndarray RGB
            else:
                # Fallback: si no expone .imgs, asumimos que cv2_img ya está anotada
                # (poco común, pero evita romper)
                annotated_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagen con detecciones")
                _render_image_safe(annotated_rgb)  # ← REEMPLAZO SEGURO (no usar st.image directo)

            # ======================
            # Tabla de objetos y conteo
            # ======================
            with col2:
                st.subheader("Objetos detectados")

                # Extraer predicciones en formato tensor/np
                predictions = None
                # Compatibilidad: distintos wrappers exponen .pred o .xyxy
                if hasattr(results, "pred") and len(results.pred) > 0:
                    predictions = results.pred[0]  # [N,6] (x1,y1,x2,y2,conf,cls)
                    boxes = predictions[:, :4]
                    scores = predictions[:, 4]
                    classes = predictions[:, 5].astype(int) if hasattr(predictions[:, 5], "astype") else predictions[:, 5].to(int)
                elif hasattr(results, "xyxy") and len(results.xyxy) > 0:
                    # Lista de tensores por imagen
                    predictions = results.xyxy[0]
                    boxes = predictions[:, :4]
                    scores = predictions[:, 4]
                    classes = predictions[:, 5].int() if hasattr(predictions[:, 5], "int") else predictions[:, 5].astype(int)
                else:
                    boxes = np.empty((0, 4))
                    scores = np.array([])
                    classes = np.array([])

                # Nombres de clases
                try:
                    label_names = results.names if hasattr(results, "names") else model.names
                except Exception:
                    label_names = getattr(model, "names", {})

                data = []
                if classes is not None and len(np.array(classes).reshape(-1)) > 0:
                    cls_np = np.array(classes).reshape(-1)
                    unique_cls, counts = np.unique(cls_np, return_counts=True)
                    for cls_id, cnt in zip(unique_cls, counts):
                        try:
                            label = label_names[int(cls_id)]
                        except Exception:
                            label = str(int(cls_id))
                        # media de confianza para esa clase
                        try:
                            mask = cls_np == int(cls_id)
                            conf_vals = np.array(scores)[mask]
                            conf_mean = float(conf_vals.mean()) if conf_vals.size > 0 else 0.0
                        except Exception:
                            conf_mean = 0.0

                        data.append(
                            {
                                "Categoría": label,
                                "Cantidad": int(cnt),
                                "Confianza promedio": f"{conf_mean:.2f}",
                            }
                        )

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("Categoría")["Cantidad"])
                else:
                    st.info("No se detectaron objetos con los parámetros actuales.")
                    st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")

        except Exception as e:
            st.error(f"Error al procesar/visualizar resultados: {e}")
            st.stop()

# ======================
# Pie de página
# ======================
st.markdown("---")
st.caption(
    "**Acerca de la aplicación**: YOLOv5 para detección de objetos, desarrollado con Streamlit y PyTorch."
)
