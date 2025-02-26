import streamlit as st
import base64
import os
from PIL import Image
import sys
from pathlib import Path
from traducciones import get_traducciones  


import os
import sys
from pprint import pprint
from dotenv import load_dotenv
import pathlib
# from tqdm import tqdm
from tqdm.autonotebook import tqdm
from pathlib import Path
from pprint import pprint
import numpy as np
import hashlib

import polars as pl
from glob import glob

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI

from PIL import Image

root_dir = Path(os.getcwd()).parent.parent
sys.path.insert(0, str(root_dir))

pl.Config.set_fmt_str_lengths(300)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20);

root_dir = Path(os.getcwd()).parent.parent
sys.path.insert(0, str(root_dir))


from src.d01_data.data import (read_json, create_index_if_not_exists, json_dump, pdf_to_images)
from src.d03_modeling.modeling import get_image_rotation, get_doc_id_from_image
from src.d00_utils.utils import get_doc_id_data #, get_doc_id_pdfs



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

client = genai.Client(api_key=GEMINI_API_KEY)
 
raw_path = root_dir / 'data' / '01_raw'
intermediate_path = root_dir / 'data' / '02_intermediate'
clean_path = root_dir / 'data' / '03_clean'
output_path = root_dir / 'data' / '04_model_output'

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


uploaded_file = st.file_uploader(traducciones[language]["subir_imagen"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 2. Guardar la imagen en un directorio local
    #    Para guardarla en el mismo directorio, puedes usar ".", 
    #    pero es más ordenado usar una carpeta como "uploads"
    save_folder = "uploads"
    # Crear la carpeta si no existe
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Crear la ruta del archivo
    file_path = os.path.join(save_folder, uploaded_file.name)

    # Guardar el archivo
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.text(file_path)

    rotation = get_image_rotation(client=client, model=GEMINI_MODEL, image_path=file_path)
    st.text(file_path)
    foto = Image.open(file_path).rotate(rotation)

    doc_id = get_doc_id_from_image(client=client, model=GEMINI_MODEL, image=foto)
    print(doc_id)
    doc_id_data = get_doc_id_data(doc_id=doc_id.codigo_documento, doc_folder=clean_path / 'formularios')




    st.image(foto, caption=traducciones[language]["imagen_subida"], use_container_width=False)    

    with st.container():
        col_pdf, col_texto = st.columns(2)
        

        formulario_key = doc_id.codigo_documento
        

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
