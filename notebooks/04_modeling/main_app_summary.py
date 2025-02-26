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

from app_image_summary_utils import *
# (
#     setup_streamlit_page,
#     select_language,
#     leer_pdf_en_base64,
#     save_uploaded_file,
#     process_image,

# )

from traducciones import get_traducciones
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI

from src.d01_data.data import read_json, create_index_if_not_exists, json_dump, pdf_to_images

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

client = genai.Client(api_key=GEMINI_API_KEY)

raw_path = root_dir / 'data' / '01_raw'
intermediate_path = root_dir / 'data' / '02_intermediate'
clean_path = root_dir / 'data' / '03_clean'
output_path = root_dir / 'data' / '04_model_output'

setup_streamlit_page()

traducciones = get_traducciones()
idiomas_opciones = {
    "Español": "español",
    "Français": "frances",
    "Deutsch": "aleman",
    "العربية": "arabe",
    "中文": "chino_mandarin"
}

language = select_language(traducciones, idiomas_opciones)
st.title(traducciones[language]["titulo"])

uploaded_file = st.file_uploader(traducciones[language]["subir_imagen"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.text(file_path)

    processed_image, doc_id, doc_id_data = process_image(file_path, client, GEMINI_MODEL, clean_path, traducciones, language)
    #st.image(processed_image, caption=traducciones[language]["imagen_subida"], use_container_width=False)

    formulario_key = doc_id.codigo_documento

    with st.container():
        col_pdf, col_texto = st.columns(2)

        with col_pdf:
            st.header(traducciones[language]["titulo_pdf"])
            pdf_traducido_path, pdf_original_path = get_pdf_paths(root_dir, formulario_key, language)
            display_pdf(traducciones, language, pdf_traducido_path, pdf_original_path)

        with col_texto:
            display_dict_as_buttons(info_dict=doc_id_data)
            
else:
    st.warning(traducciones[language]["warning_imagen"])