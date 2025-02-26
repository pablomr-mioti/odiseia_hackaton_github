import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

load_dotenv()

root_dir = Path(os.getcwd()).parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
from tqdm.autonotebook import tqdm
pl.Config.set_fmt_str_lengths(300)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)

from app_summary_utils import *

from traducciones import get_traducciones,get_key_traducciones,get_traduccion_paginas,get_traducciones_campos_imp_sel
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI

from src.d01_data.data import read_json, create_index_if_not_exists, json_dump, pdf_to_images
import src.d03_modeling.modeling as modeling
import src.d00_utils.utils as utils

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

client = genai.Client(api_key=GEMINI_API_KEY)

raw_path = root_dir / 'data' / '01_raw'
intermediate_path = root_dir / 'data' / '02_intermediate'
clean_path = root_dir / 'data' / '03_clean'
output_path = root_dir / 'data' / '04_model_output'

image_path = root_dir / 'src/d00_utils/images'

#setup_streamlit_page()
st.set_page_config(layout="wide")

image_base64 = utils.get_image_base64(image_path/'fondo-curvas.png')

# CSS con la imagen en Base64
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/png;base64,{image_base64}") no-repeat center center fixed;
    background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

traducciones = get_traducciones()

key_traducciones = get_key_traducciones()

paginas_traducciones = get_traduccion_paginas()

campos_sel_imp = get_traducciones_campos_imp_sel()

idiomas_opciones = {
    "Español": "español",
    "English": "ingles",
    "Français": "frances",
    "Deutsch": "aleman",
    "العربية": "arabe",
    "中文": "chino_mandarin",
    "Українська": "ucraniano",
}



# Función cacheada sin `client`


cols = st.columns([ 1, 5, 1])

app = cols[1]

with cols[0]:
    st.markdown(f"""
                <img style="width:140px; height:110px; margin-top: 5px; margin-bottom: 0px; margin-left: 30px;" src="data:image/png;base64,{utils.get_image_base64(f'{image_path}/odiseia-logo-white.png')}" alt="Imagen Semitransparente">
            """
            ,unsafe_allow_html=True)
    
with cols[2]:
    st.markdown(f"""
                <img style="width:170px; height:86px; margin-top: 20px; margin-bottom: 0px; margin-left: 20px;" src="data:image/png;base64,{utils.get_image_base64(f'{image_path}/mioti_data&ai_logo_white_text_final.png')}" alt="Imagen Semitransparente">
            """
            ,unsafe_allow_html=True)
    
with app:
    # Cargar imagen
    language = select_language(traducciones, idiomas_opciones)
    st.title(traducciones[language]["titulo"])
    uploaded_file = st.file_uploader(traducciones[language]["subir_imagen"], type=["png", "jpg", "jpeg"])
    language = select_language(traducciones, idiomas_opciones)
    st.title(traducciones[language]["titulo"])

    # Inicializar session_state
    if "traduced_dict" not in st.session_state:
        st.session_state.traduced_dict = None

    if "formulario_key" not in st.session_state:
        st.session_state.formulario_key = None

    if "pdf_traducido_path" not in st.session_state:
        st.session_state.pdf_traducido_path = None

    if "pdf_original_path" not in st.session_state:
        st.session_state.pdf_original_path = None

    # Función cacheada sin `client`
    @st.cache_data
    def process_image_cached(file_path, gemini_api_key, model, clean_path, traducciones, language):
        """Procesa la imagen sin recibir el cliente para evitar problemas de caché."""
        temp_client = genai.Client(api_key=gemini_api_key)  # Crear cliente dentro de la función
        return process_image(file_path, temp_client, model, clean_path, traducciones, language)

    # Función cacheada sin `client`
    @st.cache_data
    def translate_text_cached(texto, gemini_api_key, model, language):
        """Traduce texto sin recibir el cliente."""
        temp_client = genai.Client(api_key=gemini_api_key)  # Crear cliente dentro de la función
        traduced_text = modeling.gemini_translate_and_format(client=temp_client, model=model, language=language, texto_a_traducir=texto)
        return traduced_text.parsed.traduccion_formateada

    # Cargar imagen
    #uploaded_file = st.file_uploader(traducciones[language]["subir_imagen"], type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.text(file_path)

        # Procesar imagen solo si no está en session_state
        if st.session_state.traduced_dict is None:
            processed_image, doc_id, doc_id_data = process_image_cached(file_path, GEMINI_API_KEY, GEMINI_MODEL, clean_path, traducciones, language)
            
            del doc_id_data['Código del documento']
            
            campos_a_rellenar_dict = utils.parse_campos_a_rellenar(doc_id_data['Campos a rellenar'])
            
            values_to_translate = [
                ", ".join(values)  # Une todos los valores de la lista en un solo string separado por comas
                for page_data in campos_a_rellenar_dict.values()
                for values in page_data.values()
                if isinstance(values, list)
            ]

            # Verificar la cantidad de valores extraídos
            print(f"Cantidad de valores extraídos: {len(values_to_translate)}")

            # Traducir los valores
            translated_values = [translate_text_cached(value, GEMINI_API_KEY, GEMINI_MODEL, language) for value in values_to_translate]

            # Verificar la cantidad de valores traducidos
            print(f"Cantidad de valores traducidos: {len(translated_values)}")

            # Crear iterador
            translated_iter = iter(translated_values)

            # Reconstruir el diccionario con valores traducidos
            translated_dict = {}

            for page, page_data in campos_a_rellenar_dict.items():
                translated_dict[page] = {}
                for key, values in page_data.items():
                    if isinstance(values, list):
                        translated_dict[page][key] = next(translated_iter).split(", ")  # Reconstruir lista original
                    else:
                        translated_dict[page][key] = values
            
            translated_doc_id_data = {key: translate_text_cached(value, GEMINI_API_KEY, GEMINI_MODEL, language) for key, value in doc_id_data.items() if key != 'Campos a rellenar'}
            
            translated_doc_id_data['Campos a rellenar'] = translated_dict
            

            st.session_state.translated_doc_id_data = translated_doc_id_data
            st.session_state.formulario_key = doc_id.codigo_documento

            # Obtener paths de PDFs
            pdf_traducido_path, pdf_original_path = get_pdf_paths(Path(os.getcwd()).parent.parent, st.session_state.formulario_key, language)
            st.session_state.pdf_traducido_path = pdf_traducido_path
            st.session_state.pdf_original_path = pdf_original_path

        else:
            translated_doc_id_data = st.session_state.translated_doc_id_data

        # UI con columnas
        with st.container():
            col_pdf, col_texto = st.columns(2)

            with col_pdf:
                st.header(traducciones[language]["titulo_pdf"])
                display_pdf(traducciones, language, st.session_state.pdf_traducido_path, st.session_state.pdf_original_path)

            with col_texto:
                display_dict_as_dropdown(translated_doc_id_data, language,key_traducciones,paginas_traducciones,campos_sel_imp)

    else:
        st.warning(traducciones[language]["warning_imagen"])