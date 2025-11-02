import os
import sys
import io
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
# ==== ONE-PASTE FIX: forzar render seguro de imágenes en Streamlit Cloud ====
import io
from PIL import Image
import numpy as np

def _render_image_safe(img):
    """
    Acepta:
      - PIL.Image.Image
      - np.ndarray (H,W,3) en RGB o BGR
    Renderiza por bytes (PNG) para evitar el TypeError de Streamlit Cloud.
    """
    # Convertir a PIL RGB
    if isinstance(img, np.ndarray):
        # Normalizar tipo
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Si viene de OpenCV suele ser BGR -> RGB
        try:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        raise TypeError("img debe ser PIL.Image o np.ndarray HxWx3")

    # Volcar como PNG a memoria y enviar bytes a Streamlit
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    # mostramos por bytes; ignoramos kwargs como channels para evitar error
    return _st_image_original(buf.getvalue(), use_container_width=True)

# Guardar referencia original y reemplazar st.image por una versión segura
_st_image_original = st.image
def _st_image_monkey(data, *args, **kwargs):
    return _render_image_safe(data)

st.image = _st_image_monkey
# ==== FIN ONE-PASTE FIX ======================================================
