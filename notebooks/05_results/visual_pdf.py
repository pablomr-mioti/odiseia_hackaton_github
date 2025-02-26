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

idiomas_opciones = {
    "Español": "español",
    "English": "ingles",
    "Français": "frances",
    "Deutsch": "aleman",
    "العربية": "arabe",
    "中文": "chino_mandarin",
    "Українська": "ucraniano",
}




def get_pdf_paths():
    """
    Construye las rutas para el PDF traducido y el original.
    """
    pdf_traducido = os.path.abspath(
        os.path.join(
            root_dir,
            "data", "04_model_output", "formularios",
            language,
            f"{formulario_key}_{language}.pdf"
        )
    )
    pdf_original = os.path.abspath(
        os.path.join(
            root_dir,
            "data", "04_model_output", "formularios",
            "español",
            "EX-10_español.pdf"
        )
    )
    return pdf_traducido, pdf_original


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

    st.title("Visualizador de Imagen y PDF Lado a Lado")

    # Especifica las rutas de tus archivos (modifica estos valores con tus rutas reales)
    image_path = "ruta/a/tu/imagen.jpg"  # Ejemplo: "C:/imagenes/mi_imagen.jpg"
    pdf_path = "ruta/a/tu/documento.pdf"   # Ejemplo: "C:/pdfs/mi_documento.pdf"
    
    pdf_path_1 , pdf_path_2 = get_pdf_paths()

    # Creamos dos columnas para mostrar los archivos lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.header("PDF")
        try:
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al cargar el PDF: {e}")


    with col2:
        st.header("PDF")
        try:
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error al cargar el PDF: {e}")
