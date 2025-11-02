with col1:
    st.subheader("Imagen con detecciones")
    results.render()  # anota internamente
    try:
        # mostrar la imagen anotada que guarda YOLOv5
        annotated = results.imgs[0][:, :, ::-1]  # BGR->RGB si hiciste render con cv2
        st.image(annotated, use_container_width=True)
    except Exception:
        # fallback si cambia la API
        st.image(cv2_img, channels='BGR', use_container_width=True)
