import streamlit as st
import base64
import os
from PIL import Image
from traducciones import get_traducciones  # Importa la función que devuelve el diccionario de traducciones

st.set_page_config(page_title="Interfaz PDF/Foto y Texto", layout="wide")


traducciones = get_traducciones()

idiomas_opciones = {
    "Español": "español",
    "Français": "frances",
    "Deutsch": "aleman",
    "العربية": "arabe",
    "中文": "chino_mandarin"
}


if "language" not in st.session_state:
    st.session_state.language = None

if st.session_state.language is None:
    
    st.title(
        "Bienvenido - Elige tu idioma / "
        "Welcome - Choose your language / "
        "Bienvenue - Choisissez votre langue / "
        "Willkommen - Wählen Sie Ihre Sprache / "
        "مرحبًا - اختر لغتك / "
        "欢迎 - 选择你的语言")
    with st.form(key="language_form"):
        idioma_seleccionado = st.selectbox(
            "Selecciona tu idioma / Sélectionnez votre langue / Wähle deine Sprache / اختر لغتك / 选择你的语言:",
            list(idiomas_opciones.keys())
        )
        submitted = st.form_submit_button("Confirmar idioma")
        if submitted:
            st.session_state.language = idiomas_opciones[idioma_seleccionado]

    if st.session_state.language is None:
        st.stop()


language = st.session_state.language
st.title(traducciones[language]["titulo"])

def leer_pdf_en_base64(ruta_pdf):
    try:
        with open(ruta_pdf, "rb") as f:
            pdf_file = f.read()
        return base64.b64encode(pdf_file).decode("utf-8")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {ruta_pdf}")
        return None


foto = st.file_uploader(traducciones[language]["subir_imagen"], type=["png", "jpg", "jpeg"])

if foto is not None:

    imagen = Image.open(foto)
    st.image(imagen, caption=traducciones[language]["imagen_subida"], use_container_width=False)
    

    with st.container():
        col_pdf, col_texto = st.columns(2)
        

        formulario_key = 'EX-10'
        

        if "mostrar_original" not in st.session_state:
            st.session_state.mostrar_original = False

        # Columna izquierda: Visualización del PDF
        with col_pdf:
            st.header(traducciones[language]["titulo_pdf"])
            if st.button(traducciones[language]["cambiar_pdf"]):
                st.session_state.mostrar_original = not st.session_state.mostrar_original

            # Definir las rutas de los PDF (ajusta las rutas según tu estructura)
            ruta_pdf_traducido = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "data", "04_model_output", "formularios",
                   language,
                    f"{formulario_key}_{language}.pdf"
                )
            )
            ruta_pdf_original = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "data", "04_model_output", "formularios",
                    "español",
                    "EX-10_0.pdf"
                )
            )
            
            # Mostrar el PDF según la opción seleccionada
            if st.session_state.mostrar_original:
                base64_pdf = leer_pdf_en_base64(ruta_pdf_original)
                st.subheader(traducciones[language]["mostrando_pdf_original"])
            else:
                base64_pdf = leer_pdf_en_base64(ruta_pdf_traducido)
                st.subheader(traducciones[language]["mostrando_pdf_traducido"])
            
            if base64_pdf:
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

        # Columna derecha: Mostrar el contenido del archivo TXT
        with col_texto:
            st.header(traducciones[language]["contenido_txt"])
            ruta_txt = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "data", "03_clean", "formularios",
                    #language,
                    f"{formulario_key}.txt"
                )
            )
            try:
                with open(ruta_txt, "r", encoding="utf-8") as f:
                    contenido_txt = f.read()
                st.text_area(traducciones[language]["texto_area"], value=contenido_txt, height=400)
            except FileNotFoundError:
                st.error(f"No se encontró el archivo en: {ruta_txt}")
else:
    st.warning(traducciones[language]["warning_imagen"])
