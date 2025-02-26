import streamlit as st
import base64
import os
from PIL import Image


st.set_page_config(page_title="Interfaz PDF/Foto y Texto", layout="wide")
st.title("Interfaz PDF/Foto y Texto")


col_archivo, col_texto = st.columns(2)

# --- Columna Izquierda: Cargar y mostrar PDF o imagen ---
with col_archivo:
    st.header("Visualización de PDF o Imagen")
    archivo = st.file_uploader("Sube un archivo PDF o imagen", type=["pdf", "png", "jpg", "jpeg"])
    if archivo is not None:
        if archivo.type == "application/pdf":

            base64_pdf = base64.b64encode(archivo.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        elif archivo.type.startswith("image"):
            imagen = Image.open(archivo)
            st.image(imagen, caption="Imagen subida", use_container_width=True)



# --- Columna Derecha: Mostrar texto desde un archivo TXT ---
with col_texto:
    st.header("Contenido del archivo TXT")

    ruta_txt = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "03_clean", "formularios", "EX-10.txt")
    )
    try:
        with open(ruta_txt, "r", encoding="utf-8") as f:
            contenido_txt = f.read()
        st.text_area("Texto del archivo", value=contenido_txt, height=400)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo en: {ruta_txt}")
