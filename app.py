# --- Pega este bloque en app.py EN LUGAR de tu línea:
#    # ✅ Reemplazo:
_render_image_safe(img_pil)

# (Funciona tanto si tienes img_pil (PIL) como un array NumPy RGB/BGR.) 

# ==== FIX de renderizado seguro para Streamlit Cloud ====
import io
from PIL import Image
import numpy as np

def _render_image_safe(img):
    """
    Acepta:
      - PIL.Image.Image
      - np.ndarray (H,W,3) en RGB o BGR
    Renderiza por bytes para evitar el TypeError en Streamlit Cloud.
    """
    # Convertir a PIL en RGB
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        # Si viene de OpenCV probablemente es BGR: intentamos convertir a RGB
        try:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        # Asegurar modo RGB
        pil = img.convert("RGB")
    else:
        raise TypeError("img debe ser PIL.Image o np.ndarray HxWx3")

    # Volcar como PNG a memoria y pasar bytes a st.image
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf.getvalue(), use_container_width=True)
# ==== FIN FIX ====

# 👉 Llamada (reemplaza tu st.image(...)):
_render_streamlit_image(img_pil)   # o _render_streamlit_image(annotated_rgb)
