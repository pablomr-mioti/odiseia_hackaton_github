import os
import sys
from pprint import pprint
from dotenv import load_dotenv
import pathlib
import re

# from tqdm import tqdm
from tqdm.autonotebook import tqdm
from pathlib import Path
from pprint import pprint
import hashlib
import unicodedata
import tiktoken
from glob import glob

from thefuzz import process

import pymupdf
import base64

from langchain_core.documents import Document

custom_css = """
<style>

html, body, [class*="stApp"] {
    scroll-behavior: smooth;
    overflow: hidden !important;
}
[class*="block-container"] {
    overflow: hidden !important;
}

span {
    color: #0C2A4E
}
.header {
    background-color: #0B4C8E;
    color: #f9f8f7 !important;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    margin: 10px 0px;
    display: flex;
    justify-content: space-around;
    align-items: center;
    height: 100px;
}
.header span {
    color: #F9F8F7 !important;
    font-family: "Source Sans Pro", sans-serif;
    font-size: 2vw;
}
[data-testid="stHorizontalBlock"]:first-of-type span {
    color: #0C2A4E;
    font-family: "Source Sans Pro", sans-serif;
}
[data-testid="stMarkdownContainer"] p{
    font-size:calc(0.5rem + 0.5vw);
    font-weight: bold;
}
[data-testid="stMarkdownContainer"]:first-of-type{
    margin:0px;
}
[data-testid="stAppViewContainer"] {
    background-color: #f4f4f4;
    width:100%;
}
[data-testid="stAppViewBlockContainer"] {
    width:100%;
    max-width:100%;
}

span[data-baseweb="tag"] {
    background-color: #66A0CB !important;
}

div[data-baseweb="select"] > div {
    background-color: #dee8f2;
}

.st-am {
    background-color: #dee8f2;
    border-radius: 5px;
}

# .st-am.st-cz {
#     background-color: #f5f2dd;
#     border-radius: 5px;
# }

/* Contenedor de la burbuja flotante */
.bubble {
    position: fixed;
    top: 80px;
    right: 20px;
    background-color: #0c2a4e;
    color: white;
    padding: 10px 20px;
    border-radius: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    font-family: "Source Sans Pro", sans-serif;
    font-size: 12px;
    z-index: 9999;
    # cursor: pointer;
}

/* Animación de la burbuja flotante */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.bubble {
    animation: fadeIn 1s;
}

.sidebar-footer {
    text-align: center;
    font-size: 14px;
    color: #555; /* Ajusta el color del texto */
}

.heart {
    color: #66A0CB; /* Ajusta el color del emoji de corazón si es necesario */
}

# /* Estilo para el contenedor principal*/
# div.st-emotion-cache-0.e6rk8up5:nth-of-type(2) {
#     background-color: #FFFFFF !important;
#     padding: 10px;
#     border-radius: 10px;
#     padding-left: 30px;
#     padding-right: 30px;
#     padding-bottom: 30px;
#     # height: 500px; /* Ajusta la altura según necesites */
#     # overflow-y: auto;
#     # overflow-y: hidden; /* Evita el scroll en el contenedor exterior */
# }

# /* Estilo para el contenedor que tiene el historial del chat*/
# div.st-emotion-cache-b95f0i.e6rk8up4:nth-of-type(7) {
#     height: 300px; /* Ajusta la altura según necesites */
#     background-color: red;
#     # overflow-y: auto;
# }

/* Estilo para el contenedor principal (sin scroll) */
div.st-emotion-cache-0.e6rk8up5:nth-of-type(2) {
    background-color: #FFFFFF !important;
    padding: 10px;
    border-radius: 10px;
    padding-left: 30px;
    padding-right: 30px;
    padding-bottom: 30px;
    height: 550px; /* Altura fija */
    display: flex;
    flex-direction: column; /* Para alinear bien el contenido */
    justify-content: flex-start; /* Mantener el contenido arriba */
    overflow-y: auto; /* Activa el scroll interno */
}

# /* Estilo para el contenedor del input del chat */
# #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-t1wise.eht7o1d4 > div > div > div > div.stHorizontalBlock.st-emotion-cache-ocqkz7.e6rk8up0 > div.stColumn.st-emotion-cache-ozcd9e.e6rk8up2 > div > div > div > div:nth-child(2) > div > div > div:nth-child(4) {
#     background-color: blue;
#     position:relative;
#     bottom: 15px;
# }

# div.st-emotion-cache-0.e6rk8up5:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] {
#     background-color: blue;
# }


# /* Estilo para el contenedor que tiene el historial del chat (con scroll) */
# div.st-emotion-cache-b95f0i.e6rk8up4:nth-of-type(7) {
#     background-color: pink;
#     flex-grow: 1; /* Ocupará el espacio restante dentro del contenedor */
#     height: 100%; /* Ajusta la altura al padre */
#     max-height: 400px; /* Fija una altura máxima para hacer scroll */
#     overflow-y: auto; /* Activa el scroll interno */
#     padding-right: 10px; /* Evita que la barra de desplazamiento tape el contenido */
# }

</style>
"""

def get_tokens_len(text, model='gpt-4o-2024-11-20'):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def remove_duplicates(objects):
    # Initialize an empty set to store seen 'nombre_campo' values.
    seen = set()
    
    # List to store objects with unique 'nombre_campo' values.
    unique_objects = []
    
    # Iterate over each object in the provided list.
    for obj in objects:
        # Extract the 'nombre_campo' attribute from the current object.
        nombre = obj.nombre_campo
        
        # Check if this 'nombre_campo' has not been encountered before.
        if nombre not in seen:
            # Mark this 'nombre_campo' as seen by adding it to the set.
            seen.add(nombre)
            # Append the object to the unique_objects list since it's the first occurrence.
            unique_objects.append(obj)
    
    # Return the list containing objects with unique 'nombre_campo' values.
    return unique_objects


def strip_accents(text):
    """
    Remove accent characters from the given string.

    This function normalizes the string into its decomposed form (NFKD) and then
    filters out any combining characters (i.e., the accent marks).

    Args:
        text (str): The input string from which accents should be removed.

    Returns:
        str: A new string with the accents removed.
    """
    
    if text is None:
        return None
    
    # Normalize the text to NFKD form.
    normalized_text = unicodedata.normalize('NFKD', text)
    # Rebuild the string excluding the combining diacritical marks.
    stripped_text = ''.join(
        char for char in normalized_text if not unicodedata.combining(char)
    )
    return stripped_text



def metadata_to_uuid(docs):
    """
    Generate UUID-like strings from document metadata.
    Args:
        docs (list): A list of document objects. Each document is expected to have a
                     'metadata' attribute that is a dictionary.

    Returns:
        list: A list of UUID-like strings generated from the documents' metadata.
    """
    # Create a list of initial UUID strings for each document.
    # For each document, iterate over its metadata keys:
    # - Use the metadata value if the key does not start with 'Tema'.
    # - Use the key itself if it starts with 'Tema'.
    # Then join these elements with a '#' separator.
    uuids = [
        '#'.join(
            [doc.metadata[k] if not isinstance(doc.metadata[k], bool) else k for k in doc.metadata]
        )
        for doc in docs
    ]
    
    final_uuids = []  # List to store the final UUID-like strings.
    
    # Process each initial UUID string.
    for _id in uuids:
        # Split the initial string at the first '#' into a list of parts.
        parts = _id.split('#', maxsplit=2)
        # Compute the SHA-256 hash of the second part (after encoding to UTF-8).
        hash_part = hashlib.sha256(parts[-1].encode('utf-8')).hexdigest()
        # Combine the first part and the hash (separated by '#') to form the final UUID.
        final_uuids.append(f'{strip_accents(parts[0])}#{strip_accents(parts[1])}#{hash_part}')
    
    # Return the list of final UUID-like strings.
    return final_uuids


def generate_sparse_vector_in_batches(documents, embedding_model, fitted_bm25, batch_size=64):
    """
    Generate a list of vectors that combine sparse and dense embeddings for a set of documents
    in batches.

    This function processes a list of documents in user-specified batch sizes, generating
    unique IDs for each document, extracting metadata, and creating both sparse and dense
    representations of the documents using a provided embedding model and a fitted BM25 model.
    It then combines the generated values into a dictionary format.

    Args:
        documents (list): A list of documents, each with attributes `metadata` and `page_content`.
        embedding_model: An embedding model instance with an `embed_documents` method to generate dense embeddings.
        fitted_bm25: A BM25 model instance with an `encode_documents` method to generate sparse embeddings.
        batch_size (int, optional): Number of documents to process in each batch.

    Returns:
        list: A list of dictionaries, where each dictionary contains the following keys:
              'id' (str): A unique identifier for the document.
              'sparse_values' (array-like): Sparse embedding values for the document.
              'values' (array-like): Dense embedding values for the document.
              'metadata' (dict): Metadata associated with the document.
    """

    # Generate a unique UUID for each document
    uuids = metadata_to_uuid(documents)
    
    # Add page content to metadata
    for doc in documents:
        doc.metadata['text'] = doc.page_content

    # Prepare a list to accumulate all results
    vectors = []

    # Process documents in batches
    for start_idx in range(0, len(documents), batch_size):
        end_idx = start_idx + batch_size
        batch_docs = documents[start_idx:end_idx]
        batch_uuids = uuids[start_idx:end_idx]

        # Extract metadata for the batch
        batch_metadatas = [doc.metadata for doc in batch_docs]

        # Extract page_content for the batch
        batch_contents = [doc.page_content for doc in batch_docs]

        # Generate dense embeddings for the batch
        dense_embeds = embedding_model.embed_documents(batch_contents)

        # Generate sparse embeddings using the fitted BM25 model for the batch
        sparse_values = fitted_bm25.encode_documents(batch_contents)

        # Combine the ID, sparse embedding, dense embedding, and metadata for each document in the batch
        for _id, sparse, dense, metadata in zip(batch_uuids, sparse_values, dense_embeds, batch_metadatas):
            vectors.append({
                'id': _id,
                'sparse_values': sparse,
                'values': dense,
                'metadata': metadata,
            })

    # Return the list of combined vectors
    return vectors



def dict_to_document_boe(estructured_dict, tema, origen):
    """
    Converts a structured dictionary into a list of document objects with metadata.

    Args:
        estructured_dict (dict): A dictionary where keys are article identifiers and values are article texts.
        tema (str): The topic or theme associated with the documents.
        origen (str): The source or origin of the documents.

    Returns:
        list: A list of Document objects, each containing the article text and its associated metadata.
    """
    
    documents = []  # Initialize an empty list to store document objects.
    for articulo, texto in estructured_dict.items():
        # Split the article text into lines.
        text_split = texto['chunk_content'].split('\n') if tema == 'orientacion' else texto.split('\n')
        
        if tema == 'orientacion':
            fichero = text_split[0].strip()
            titulo = 'TÍTULO SIN ASIGNAR'
            capitulo ='CAPÍTULO SIN ASIGNAR'
            seccion = text_split[1].strip()
            articulo = articulo = f'{seccion}_{articulo}'
        
        elif Path(origen).stem[-3:] == 'boe':
            # Extract title, chapter, and section from specific lines.
            fichero = text_split[0].strip()
            titulo = text_split[1].strip()
            capitulo = text_split[2].strip()
            seccion = text_split[3].strip()
        else:
            fichero = text_split[0].strip()
            titulo = 'TÍTULO SIN ASIGNAR'
            capitulo ='CAPÍTULO SIN ASIGNAR'
            seccion = 'SECCIÓN SIN ASIGNAR'
            articulo = f'{Path(origen).stem}_{articulo}'
                
        # Format the full text for the document.
        texto = f'{tema}\n{texto}'
        
         # Prepare metadata for the document.
        metadata = {
            'tema': tema,
            'fichero': fichero,
            'titulo': titulo,
            'capitulo': capitulo,
            'seccion': seccion,
            'articulo': articulo}

        # Create a Document object with the text and metadata.
        document = Document(
            page_content=texto,
            metadata=metadata
        )
        # Append the document to the list.
        documents.append(document)
        
    return documents



def split_pdf_into_pages(pdf_path, output_path):
    """
    Splits a PDF file into individual pages and saves each page as a separate PDF file.

    Parameters:
        pdf_path (str or Path): The path to the input PDF file.
        output_path (str or Path): The directory where the split PDF files will be saved.
    """
    # Convert the provided paths to pathlib.Path objects for easier path manipulation.
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    # Open the original PDF document.
    doc = pymupdf.open(pdf_path)
    # Determine the total number of pages in the PDF.
    n_pages = len(doc)
    # Create a list of page indices (e.g., [0, 1, 2, ..., n_pages-1]).
    n_pages_list = list(range(n_pages))

    # Iterate over each page index to extract and save that page.
    for i in range(n_pages):
        # Create a new, empty PDF document to hold the current page.
        page_pdf = pymupdf.open()

        # For pages after the first one, re-open the original PDF and reinitialize the page list.
        if i != 0:
            doc = pymupdf.open(pdf_path)
            n_pages_list = list(range(n_pages))

        # Remove the current page index from the list to keep only the desired page.
        n_pages_list.remove(i)
        # Delete all pages from the document except the one we want to extract.
        doc.delete_pages(n_pages_list)

        # Insert the remaining page into the new PDF document at position 0.
        page_pdf.insert_pdf(doc, to_page=0)

        # Construct the output file path:
        # The file will be saved in a subdirectory named after the PDF file's stem,
        # and the filename will include the PDF stem and the current page index.
        file_output_path = output_path / Path(f'{pdf_path.stem}/{pdf_path.stem}_{str(i)}.pdf')
        # Ensure that the directory for the output file exists; create it if necessary.
        os.makedirs(file_output_path.parent, exist_ok=True)

        # Save the single-page PDF to the constructed file path.
        page_pdf.save(file_output_path)
        # Print a confirmation message indicating where the file was saved.
        print(f'File saved in {file_output_path}')
        
        
        

def parse_document_from_file(filepath):
    """BlockingIOError
    Reads a document from a file and returns a dictionary with the following keys:
    - 'Código del documento'
    - 'Leyes en las que sustentan'
    - 'Procesos donde este formulario es necesario'
    - 'Instrucciones de como rellenar el documento'
    - 'Secciones a rellenar del formulario'
    
    Each key's value is the text content of that section.
    """
    # Read the content of the file.
    with open(filepath, 'r', encoding='utf-8') as file:
        long_string = file.read()
    
    # Define the section headers to look for (order is important)
    sections = [
        'Resumen del documento',
        'Código del documento',
        'Leyes en las que sustentan',
        'Procesos donde este formulario es necesario',
        'Instrucciones de como rellenar el documento',
        'Secciones a rellenar del formulario',
        'Campos a rellenar'
    ]
    
    extracted = {}
    
    # Iterate over each section and capture the content until the next section header or the end of the string.
    for i, header in enumerate(sections):
        # Build a regex pattern for the current section.
        next_headers = sections[i+1:]
        if next_headers:
            # Lookahead pattern: stops when any of the next headers (followed by a colon) is encountered.
            lookahead = r'(?=' + '|'.join(re.escape(h) + r':' for h in next_headers) + r')'
        else:
            # For the last section, capture until the end of the string.
            lookahead = r'(?=\Z)'
        
        pattern = re.escape(header) + r':\s*(.*?)\s*' + lookahead
        
        # Search the pattern in the long string with DOTALL flag to include newlines.
        match = re.search(pattern, long_string, re.DOTALL)
        if match:
            extracted[header] = match.group(1).strip()
        else:
            extracted[header] = ''
    
    return extracted

def get_doc_id_data(doc_id, doc_folder):
    """
    Given a document ID and a folder path, returns its contents using the
    parse_document_from_file function.
    
    Parameters:
    - doc_id (str): The unique identifier for the document.
    - doc_folder (str or Path): The folder where the document file is stored.
    
    Returns:
    - dict or None: A dictionary with the parsed sections of the document if found; otherwise, None.
    """
    # Convert the provided folder path into a Path object for easier path manipulations.
    doc_folder = Path(doc_folder)
    
    # Construct the file path using the document ID.
    # It assumes that the file is named as <doc_id>.txt inside the doc_folder.
    id_folder_path = Path(f'{doc_folder / doc_id}.txt')
    
    # Check if the file exists at the constructed path.
    if id_folder_path.exists():
        # If the file exists, parse its content and return the parsed data.
        return parse_document_from_file(str(id_folder_path))
    else:
        # If the file does not exist, print an error message and return None.
        print(f'No file found for the code "{doc_id}"')
        return None


    
def campos_to_srt(campos_obj, imputados):
    
    result_string = []
    
    for page in campos_obj.keys():
        result_string.append(f'pagina {page}')
        if imputados:
            result_string.append('Campos de imputacion')
        else:
            result_string.append('Campos de selección')
        
        for obj in campos_obj[page]:
            obj_string = f'Nombre: {obj.nombre_campo}.'
            # if imputados and obj.nombre_completo:
            #     obj_string += f' Nombre completo: {obj.nombre_completo}'
            result_string.append(obj_string)

    return '\n'.join(result_string)


def campos_text_dict(text):
    resultado = {}
    pagina_actual = None
    categoria_actual = None

    # Recorremos el texto línea a línea
    for linea in text.splitlines():
        linea = linea.strip()
        if not linea:
            continue

        # Si la línea indica una página, extraemos la cadena completa 'pagina X'
        if linea.lower().startswith('pagina'):
            # Se espera un formato 'pagina X'
            partes = linea.split()
            if len(partes) >= 2:
                pagina_actual = 'pagina ' + partes[1]
                # Si la página ya existe, se mantiene la información previa
                if pagina_actual not in resultado:
                    resultado[pagina_actual] = {
                        'Campos de imputacion': [],
                        'Campos de selección': []
                    }
            continue

        # Detectamos la categoría
        if linea.lower().startswith('campos de imputacion'):
            categoria_actual = 'Campos de imputacion'
            continue

        if linea.lower().startswith('campos de selección'):
            categoria_actual = 'Campos de selección'
            continue

        # Procesamos las líneas de campo que empiezan por 'Nombre:'
        if linea.startswith('Nombre:'):
            # Eliminamos la parte 'Nombre:' y limpiamos espacios
            campo = linea[len('Nombre:'):].strip()
            # Opcional: quitar el punto final si lo tiene
            if campo.endswith('.'):
                campo = campo[:-1].strip()

            # Agregamos el campo a la categoría y página correspondiente
            if pagina_actual is not None and categoria_actual is not None:
                resultado[pagina_actual][categoria_actual].append(campo)
            continue

    return resultado


def normalize_string(text):
    # Split the text into lines
    lines = text.splitlines()
    words = []
    
    for line in lines:
        # Remove leading/trailing whitespace
        stripped_line = line.strip()
        # Skip empty lines or lines that contain only dashes
        if not stripped_line or all(ch == '-' for ch in stripped_line):
            continue
        
        # Check if the line contains a sequence of three or more spaces
        if re.search(r'\s{3,}', stripped_line):
            # Split the line using sequences of 3+ whitespace characters
            parts = re.split(r'\s{3,}', stripped_line)
            for part in parts:
                cleaned = part.strip().rstrip('.')
                if cleaned:
                    words.append(cleaned)
        else:
            cleaned = stripped_line.rstrip('.')
            words.append(cleaned)
    
    return words


def extract_section_names(text):
    """
    Extrae de un texto los nombres de las secciones.
    
    Busca líneas que comiencen con "Sección" y captura el contenido que sigue
    """
    # Patrón que busca líneas que empiecen con "Sección", seguido de dígitos y ')'
    pattern = r"^Sección\s+(\d\).+)$"
    # re.M para que ^ actúe al inicio de cada línea
    matches = re.findall(pattern, text, re.M)
    # Retornamos cada resultado limpio de espacios laterales
    return [match.strip() for match in matches]


def match_words(word, candidates, score_cutoff=80):
    
    lower_candidates = [t.lower() for t in candidates]
    match = process.extractOne(word.lower(), lower_candidates, score_cutoff=score_cutoff)
    
    if match:
        return candidates[lower_candidates.index(match[0])]
    else:
        print(f'No match for word {word}')
        return
    
    
def remove_number_parenthesis_space(text):
    """
    Removes any pattern of one or more digits followed by a ')' and a space from the input string.
    """
    
    # Remove the "□" symbol.
    # This regex matches one or more digits followed by a ')' and a space 
    # but only if it is immediately followed by a letter.
    text = re.sub(r'□', '', text)
    return  re.sub(r'\d+\) (?=[A-Za-z])', '', text)

def clean_translation(text):
    """
    Elimina el símbolo '□' y secuencias de puntos (o elipsis) consecutivos (más de 2) del texto.
    """
    # Elimina el símbolo "□"
    text = re.sub(r'□', '', text)
    # Elimina secuencias de puntos o elipsis (ASCII . o Unicode …) de tres o más repeticiones
    text = re.sub(r'[.\u2026]{3,}', '', text)
    # Elimina espacios extra al inicio y final del texto
    return text.strip()

def filter_rectangles_by_area(rectangles, min_area):
    """
    Filters a list of rectangle objects (e.g. fitz.Rect) returning only those
    whose area is greater than or equal to min_area.

    The function attempts to use the rectangle's getArea() method if available;
    otherwise, it calculates the area using width and height properties.
    
    Parameters:
        rectangles (list): A list of rectangle objects.
        min_area (float): The minimum area threshold.

    Returns:
        list: A list of rectangle objects with area >= min_area.
    """
    filtered = []
    
    for rect in rectangles:
        # Use getArea() if available, otherwise calculate using width and height.
        if hasattr(rect, "getArea"):
            area = rect.getArea()
        else:
            area = rect.width * rect.height
        if area >= min_area:
            filtered.append(rect)
    
    return filtered
    
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        src_base64 = base64.b64encode(image_file.read()).decode()
    return src_base64


def parse_campos_a_rellenar(input_text):
    resultado = {}
    pagina_actual = None
    campo_tipo_actual = None

    # Iteramos por cada línea del texto
    for linea in input_text.splitlines():
        linea = linea.strip()
        if not linea:
            continue

        # Detectamos el cambio de página
        if linea.lower().startswith('pagina'):
            pagina_actual = linea
            if pagina_actual not in resultado:
                resultado[pagina_actual] = {}
        # Detectamos si se trata de un bloque de 'Campos de imputacion' o 'Campos de selección'
        elif linea.lower().startswith('campos de'):
            campo_tipo_actual = linea
            if campo_tipo_actual not in resultado[pagina_actual]:
                resultado[pagina_actual][campo_tipo_actual] = []
        # Procesamos cada línea que contenga 'Nombre:'
        elif linea.startswith('Nombre:'):
            # Extraemos el valor posterior a 'Nombre:'
            valor = linea[len('Nombre:'):].strip()
            # Opcionalmente eliminamos el punto final si existe
            if valor.endswith('.'):
                valor = valor[:-1].strip()
            resultado[pagina_actual][campo_tipo_actual].append(valor)
    
    return resultado

def get_content_languages_dict():
    content_languages_dict = {
        "en": {
            "Record a voice message": "Record a voice message",
            "Transcribing and sending...": "Transcribing and sending...",
            "Loading language model...": "Loading language model...",
            "🌐 Language selected: ": "🌐 Language selected: ",
            "🔊 Answer": "🔊 Read the answer out loud",
            "How can I help you?": "Hello, I am an immigration assistant for the government of Spain. I am here to help you obtain a residence and work permit for exceptional circumstances. To begin, can you tell me your country of origin?",
            "Response Modes": ["Text", "Audio"],
            "Your message": "Your message"
        },
        "es": {
            "Record a voice message": "Grabar un mensaje de voz",
            "Transcribing and sending...": "Transcribiendo y enviando...",
            "Loading language model...": "Cargando modelo de lenguaje...",
            "🌐 Language selected: ": "🌐 Idioma seleccionado: ",
            "🔊 Answer": "🔊 Leer la respuesta en voz alta",
            "How can I help you?": "Hola, soy un asistente de inmigración del gobierno de España. Estoy aquí para ayudarte a obtener un permiso de residencia y trabajo por circunstancias excepcionales. Para comenzar, ¿puedes decirme tu país de origen?",
            "Response Modes": ["Texto", "Audio"],
            "Your message": "Tu mensaje"
        },
        "fr": {
            "Record a voice message": "Enregistrer un message vocal",
            "Transcribing and sending...": "Transcription et envoi...",
            "Loading language model...": "Chargement du modèle de langue...",
            "🌐 Language selected: ": "🌐 Langue sélectionnée : ",
            "🔊 Answer": "🔊 Lire la réponse à voix haute",
            "How can I help you?": "Bonjour, je suis un assistant d'immigration du gouvernement espagnol. Je suis ici pour vous aider à obtenir un permis de séjour et de travail pour des circonstances exceptionnelles. Pour commencer, pouvez-vous me dire votre pays d'origine ?",
            "Response Modes": ["Texte", "Audio"],
            "Your message": "Votre message"
        },
        "de": {
            "Record a voice message": "Eine Sprachnachricht aufnehmen",
            "Transcribing and sending...": "Transkribieren und senden...",
            "Loading language model...": "Sprachmodell wird geladen...",
            "🌐 Language selected: ": "🌐 Sprache ausgewählt: ",
            "🔊 Answer": "🔊 Die Antwort laut vorlesen",
            "How can I help you?": "Hallo, ich bin ein Einwanderungsassistent der spanischen Regierung. Ich bin hier, um Ihnen zu helfen, eine Aufenthalts- und Arbeitserlaubnis aus außergewöhnlichen Gründen zu erhalten. Können Sie mir zunächst Ihr Herkunftsland nennen?",
            "Response Modes": ["Text", "Audio"],
            "Your message": "Ihre Nachricht"
        },
        "ar": {
            "Record a voice message": "تسجيل رسالة صوتية",
            "Transcribing and sending...": "جارٍ النسخ والإرسال...",
            "Loading language model...": "جارٍ تحميل نموذج اللغة...",
            "🌐 Language selected: ": "🌐 اللغة المختارة: ",
            "🔊 Answer": "🔊 اقرأ الإجابة بصوت عالٍ",
            "How can I help you?": "مرحبًا، أنا مساعد الهجرة التابع للحكومة الإسبانية. أنا هنا لمساعدتك في الحصول على تصريح إقامة وعمل لظروف استثنائية. للبدء، هل يمكنك إخباري ببلدك الأصلي؟",
            "Response Modes": ["نص", "صوت"],
            "Your message": "رسالتك"
        },
        "zh": {
            "Record a voice message": "录制语音消息",
            "Transcribing and sending...": "正在转录并发送...",
            "Loading language model...": "正在加载语言模型...",
            "🌐 Language selected: ": "🌐 选择的语言：",
            "🔊 Answer": "🔊 大声朗读答案",
            "How can I help you?": "你好，我是西班牙政府的移民助手。我在这里帮助您申请特殊情况的居留和工作许可。首先，您能告诉我您的原籍国吗？",
            "Response Modes": ["文本", "音频"],
            "Your message": "您的消息"
        },
        "uk": {
            "Record a voice message": "Записати голосове повідомлення",
            "Transcribing and sending...": "Транскрибування та надсилання...",
            "Loading language model...": "Завантаження мовної моделі...",
            "🌐 Language selected: ": "🌐 Вибрана мова: ",
            "🔊 Answer": "🔊 Прочитайте відповідь вголос",
            "How can I help you?": "Привіт, я помічник з питань імміграції уряду Іспанії. Я тут, щоб допомогти вам отримати дозвіл на проживання та роботу за виняткових обставин. Спочатку, чи можете ви сказати мені вашу країну походження?",
            "Response Modes": ["Текст", "Аудіо"],
            "Your message": "Ваше повідомлення"
        },
    }

    return content_languages_dict