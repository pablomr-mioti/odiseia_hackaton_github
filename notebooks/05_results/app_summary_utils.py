import os
import base64
from pathlib import Path
from PIL import Image
import streamlit as st

from traducciones import get_traducciones

translations = get_traducciones() 

def setup_streamlit_page():
    """Configura la página de Streamlit."""
    st.set_page_config(page_title="Interfaz PDF/Foto y Texto", layout="wide")

def select_language(traducciones, idiomas_opciones):
    """
    Muestra el formulario de selección de idioma.
    Si aún no se ha elegido, detiene la ejecución.
    """
    if "language" not in st.session_state:
        st.session_state.language = None

    if st.session_state.language is None:
        st.title(
            "Bienvenido - Elige tu idioma / Welcome - Choose your language / "
            "Bienvenue - Choisissez votre langue / Willkommen - Wählen Sie Ihre Sprache / "
            "مرحبًا - اختر لغتك / 欢迎 - 选择你的语言 / Ласкаво просимо - виберіть свою мову"
        )
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
    return st.session_state.language

def leer_pdf_en_base64(ruta_pdf):
    """
    Lee un archivo PDF y lo codifica en base64 para poder incrustarlo en un iframe.
    """
    try:
        with open(ruta_pdf, "rb") as f:
            pdf_file = f.read()
        return base64.b64encode(pdf_file).decode("utf-8")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {ruta_pdf}")
        return None

def save_uploaded_file(uploaded_file, save_folder="uploads"):
    """
    Guarda el archivo subido en la carpeta indicada y retorna la ruta donde se guardó.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_path = os.path.join(save_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_image(file_path, client, GEMINI_MODEL, clean_path, traducciones, language):
    """
    Procesa la imagen:
      - Obtiene la rotación correcta.
      - Rota la imagen.
      - Obtiene el ID del documento a partir de la imagen.
      - Si no se obtiene un doc_id, muestra un warning usando la traducción correspondiente.
    
    Retorna la imagen procesada, el objeto doc_id, y los datos doc_id_data.
    """
    from src.d03_modeling.modeling import get_image_rotation, get_doc_id_from_image
    from src.d00_utils.utils import get_doc_id_data
    rotation = get_image_rotation(client=client, model=GEMINI_MODEL, image_path=file_path)

    processed_image = Image.open(file_path).rotate(rotation)
    doc_id = get_doc_id_from_image(client=client, model=GEMINI_MODEL, image=processed_image)

    if doc_id is None:
        st.warning(traducciones[language]["sube_otra_foto_warning"])
        return processed_image, None, None

    doc_id_data = get_doc_id_data(
        doc_id=doc_id.codigo_documento,
        doc_folder=clean_path / 'formularios'
    )
    
    return processed_image, doc_id, doc_id_data

def get_pdf_paths(root_dir, formulario_key, language):
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

def display_dict_as_buttons(info_dict):
    """
    Muestra un grupo de botones (uno por cada clave del diccionario).
    Al hacer clic en un botón, se muestra el contenido asociado a esa clave.

    Parámetros:
    -----------
    info_dict : dict
        Diccionario cuyas claves se mostrarán como botones.
        Los valores son los textos que se mostrarán al hacer clic.
    """

    # Si no existe en session_state, inicializamos la clave para el elemento seleccionado
    if "selected_dict_key" not in st.session_state:
        st.session_state["selected_dict_key"] = list(info_dict.keys())[0]

    st.write("### Seleccione una sección para ver el contenido:")

    # 1) Zona de botones (uno para cada clave)
    for key in info_dict:
        if st.button(key):
            st.session_state["selected_dict_key"] = key

    st.write("---")

    # 2) Muestra el contenido asociado a la clave seleccionada
    selected_key = st.session_state["selected_dict_key"]
    if selected_key is not None:
        st.subheader(f"**Clave:** {selected_key}")
        st.text_area(
            label="",
            value=info_dict[selected_key],
            height=300
        )
    else:
        st.info("Haz clic en uno de los botones de arriba para ver su contenido.")
        
# def display_dict_as_dropdown(info_dict, language,key_traducciones,paginas_traducciones,campos_sel_imp):
#     """
#     Muestra un desplegable en la zona izquierda para seleccionar una clave del diccionario.
#     Si la opción seleccionada es "Campos a rellenar", muestra subdesplegables para páginas y tipos de campo.

#     Parámetros:
#     -----------
#     info_dict : dict
#         Diccionario cuyas claves se mostrarán en el desplegable.
#     language : str
#         Idioma actual para las traducciones.
#     """
#     st.write(f"### {translations[language]['seleccione_una_seccion']}")
    
#     key_traducciones_map = key_traducciones[language]
    
#     info_dict = {key_traducciones_map.get(old_key, old_key): value for old_key, value in info_dict.items()}    

#     campos_rellenar_translated = key_traducciones[language].get('Campos a rellenar')
    
#     print(' campos a rellenar translated',campos_rellenar_translated)
#     print(type(campos_rellenar_translated))

#     selected_key = st.selectbox(f"{translations[language]['selecciona']}", list(info_dict.keys()))

#     if selected_key == campos_rellenar_translated:
        
#         selected_page = st.selectbox(
#             translations[language]['paginas'],            
            
#             list(info_dict[campos_rellenar_translated].keys()))

#         selected_field_type = st.selectbox(
#             translations[language]['seleccione_tipo_campo'], 
#             list(info_dict[campos_rellenar_translated][selected_page].keys()))
        
#         st.subheader(f"{selected_field_type} en {selected_page}")
#         st.markdown(body="\n".join(info_dict[campos_rellenar_translated][selected_page][selected_field_type])
#             ,unsafe_allow_html=False
#         )

#     else:
#         st.subheader(selected_key)
#         st.markdown(body=info_dict[selected_key],unsafe_allow_html=False)



def display_dict_as_dropdown(info_dict, language,key_traducciones,paginas_traducciones,campos_sel_imp):
    """
    Muestra un desplegable en la zona izquierda para seleccionar una clave del diccionario.
    Si la opción seleccionada es "Campos a rellenar", muestra subdesplegables para páginas y tipos de campo.

    Parámetros:
    -----------
    info_dict : dict
        Diccionario cuyas claves se mostrarán en el desplegable.
    language : str
        Idioma actual para las traducciones.
    """
    st.write(f"### {translations[language]['seleccione_una_seccion']}")
    
    key_traducciones_map = key_traducciones[language]
    campos_sel_imp = campos_sel_imp[language]
    paginas_traducciones = paginas_traducciones[language]
    
    info_dict = {key_traducciones_map.get(old_key, old_key): value for old_key, value in info_dict.items()}    

    campos_rellenar_translated = key_traducciones[language].get('Campos a rellenar')
    
    print(' campos a rellenar translated',campos_rellenar_translated)
    print(type(campos_rellenar_translated))

    selected_key = st.selectbox(f"{translations[language]['selecciona']}", list(info_dict.keys()))

    if selected_key == campos_rellenar_translated:        
        # Mapeamos las claves originales a sus traducciones para mostrarlas en el selector
        info_dict_paginas = {paginas_traducciones.get(old_key, old_key): value 
                            for old_key, value in info_dict[campos_rellenar_translated].items()}
        print('info_dict_paginas', info_dict_paginas)
        
        # Selector para páginas (mostrando la traducción)
        selected_page = st.selectbox(
            translations[language]['paginas'],            
            list(info_dict_paginas.keys())
        )
        
        # Revertir la traducción para obtener la clave original de la página
        reverse_paginas = {v: k for k, v in paginas_traducciones.items()}
        original_page = reverse_paginas.get(selected_page, selected_page)
        
        ### Campos imputación y selección
        info_dict_sel_imp = {campos_sel_imp.get(old_key, old_key): value 
                            for old_key, value in info_dict[campos_rellenar_translated][original_page].items()}
        print('info_dict_sel_imp', info_dict_sel_imp)
        
        # Selector para tipo de campo (mostrando la traducción)
        selected_field_type = st.selectbox(
            translations[language]['seleccione_tipo_campo'], 
            list(info_dict_sel_imp.keys())
        )
        
        # Revertir la traducción para obtener la clave original del campo
        reverse_campos_sel_imp = {v: k for k, v in campos_sel_imp.items()}
        original_field_type = reverse_campos_sel_imp.get(selected_field_type, selected_field_type)
        
        st.subheader(f"{selected_field_type} en {selected_page}")
        
        # Utiliza las claves originales (original_page y original_field_type) para filtrar el diccionario original
        st.markdown(
            body="\n".join(info_dict[campos_rellenar_translated][original_page][original_field_type]),
            unsafe_allow_html=False
        )   
    else:
        st.subheader(selected_key)
        st.markdown(body=info_dict[selected_key],unsafe_allow_html=False)


def display_pdf(traducciones, language, pdf_traducido_path, pdf_original_path):
    """
    Muestra el PDF traducido o el original en función de la opción elegida.
    """
    if "mostrar_original" not in st.session_state:
        st.session_state.mostrar_original = False

    if st.button(traducciones[language]["cambiar_pdf"]):
        st.session_state.mostrar_original = not st.session_state.mostrar_original

    if st.session_state.mostrar_original:
        base64_pdf = leer_pdf_en_base64(pdf_original_path)
        st.subheader(traducciones[language]["mostrando_pdf_original"])
    else:
        base64_pdf = leer_pdf_en_base64(pdf_traducido_path)
        st.subheader(traducciones[language]["mostrando_pdf_traducido"])

    if base64_pdf:
        pdf_display = f'<object data="data:application/pdf;base64,{base64_pdf}" width="600" height="1000" type="application/pdf"></object>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def display_text_content(traducciones, language, txt_file_path):
    """
    Lee y muestra el contenido del archivo TXT en un área de texto.
    """
    st.header(traducciones[language]["contenido_txt"])
    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            contenido_txt = f.read()
        st.text_area(traducciones[language]["texto_area"], value=contenido_txt, height=400)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo en: {txt_file_path}")