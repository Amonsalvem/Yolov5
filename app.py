# --- Pega este bloque en app.py EN LUGAR de tu línea:
#     st.image(img_pil, use_container_width=True)
# (Funciona tanto si tienes img_pil (PIL) como un array NumPy RGB/BGR.) 

import io
from PIL import Image
import numpy as np

def _render_streamlit_image(img):
    """
    Acepta:
      - PIL.Image.Image
      - np.ndarray (H,W,3) en RGB o BGR
    Muestra por bytes para evitar el TypeError de Streamlit Cloud.
    """
    if isinstance(img, np.ndarray):
        # Si llega BGR (cv2), conviértelo a RGB; si ya es RGB, no pasa nada.
        try:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img
    else:
        raise TypeError("img debe ser PIL.Image o np.ndarray HxWx3")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf.getvalue(), use_container_width=True)

# 👉 Llamada (reemplaza tu st.image(...)):
_render_streamlit_image(img_pil)   # o _render_streamlit_image(annotated_rgb)
