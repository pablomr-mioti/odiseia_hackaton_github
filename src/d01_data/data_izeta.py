import os
import sys
import glob
import io
import re
from pprint import pprint
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pprint import pprint
import aiofiles
import xml.etree.ElementTree as ET
 
import polars as pl
import json

import fitz
from PIL import Image
import pytesseract
import asyncio

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LTChar
from pdfminer.pdfparser import PDFSyntaxError

import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time


def read_json(filename):
    """
    Lee un archivo JSON y devuelve su contenido.
    
    Parámetros:
    filename (str): Ruta del archivo JSON a leer.

    Retorna:
    dict o list: Contenido del archivo JSON como un diccionario o lista.
    """
    try:
        with open(filename, 'r') as archivo:
            contenido = json.load(archivo)
        return contenido
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{filename}' no es un JSON válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        

async def read_json_async(filename):
    """
    Lee un archivo JSON y devuelve su contenido de manera asíncrona.

    Parámetros:
    filename (str): Ruta del archivo JSON a leer.

    Retorna:
    dict o list: Contenido del archivo JSON como un diccionario o lista.
    """
    try:
        async with aiofiles.open(filename, 'r') as archivo:
            contenido = await archivo.read()
        return json.loads(contenido)
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{filename}' no es un JSON válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

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

def get_all_id_prefixes(index):
    """
    Extracts all unique ID prefixes from a given index.
    Args:
        index: A Pinecone index
    Returns:
        set: A set containing unique ID prefixes.
    """

    # Initialize an empty set to store unique prefixes
    prefixes_set = set()

    # Iterate through all IDs in the index
    for ids in index.list():
        # Extract prefixes from each ID and add them to a temporary set
        prefixes = {_id.split('#')[0] for _id in ids}
        # Merge the temporary set with the main prefix set
        prefixes_set |= prefixes 

    # Print the number of unique prefixes found (in Spanish)
    print(f'Se ha encontrado {len(prefixes_set)} en el índice')

    return prefixes_set

def match_prefix(index, prefixes):
    """
    Matches a set of given prefixes with those found in the provided index.

    Args:
        index: index: A Pinecone index
        prefixes (iterable): A collection of prefix strings to be matched.

    Returns:
        set: A set containing prefixes that exist in both `index` and `prefixes`.
    """

    # Retrieve all unique ID prefixes from the given index
    index_prefixes = get_all_id_prefixes(index=index)

    # Normalize the input prefixes by removing accents
    prefixes = {strip_accents(pref) for pref in prefixes}

    # Return the intersection of index prefixes and normalized input prefixes
    return index_prefixes & prefixes


 
def delete_docs_by_id(index, prefix):
    """
    Delete documents from the index based on a given prefix.

    Args:
        index: A Pinecone index
        prefix (str, optional): A string prefix used to filter document IDs.

    Returns:
        list of deletd docs

    Side Effects:
        - Deletes documents from the index.
        - Prints the total number of documents removed.
    """
    docs_removed = []  # List to store the IDs of removed documents.
    
    origen = strip_accents(prefix)
    
    if origen is not None:
        # Iterate over each document ID returned by index.list.
        # The prefix is processed by strip_accents to remove any accent marks.
        for ids in index.list(prefix=origen+'#'):
            docs_removed.extend(ids) 
            index.delete(ids=ids)     # Delete the document with this ID from the index.

    print(f'Se han eliminado un total de {len(docs_removed)} documentos con el prefijo {origen}')
    
    return docs_removed

        
        
def json_dump(datos, nombre_archivo):
    """
    Guarda un objeto JSON en un archivo .json.

    Parámetros:
    datos (dict): El objeto JSON que deseas guardar.
    nombre_archivo (str): El nombre del archivo sin extensión, se añadirá ".json" automáticamente.
    """
    try:
        # Guarda el archivo con la extensión .json
        with open(f"{nombre_archivo}.json", "w") as archivo:
            json.dump(datos, archivo, indent=4)
        print(f"Archivo guardado exitosamente como {nombre_archivo}.json")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")


def create_db_if_not_exists(client, db_name):
    """
    Creates a vector store if it does not exist.
    
    Parameters:
    db_name (str): Name of the vector store.
    
    Returns:
    vector_store: The newly created vector store or the existing one.
    """
    # Get a list of all available vector stores.
    vector_stores = client.beta.vector_stores.list().data
    vec_names = [db.name for db in vector_stores]  # Extract the names of the vector stores.

    # If the database name is not in the list, create a new vector store.
    if db_name not in vec_names:
        vector_store = client.beta.vector_stores.create(name=db_name)
        print(f'Vector store {db_name} created')
        return vector_store
    else:
        # If the vector store already exists, print a message and retrieve it.
        print(f'The vector store {db_name} already exists')
        for db in vector_stores:
            if db.name == db_name:
                # Retrieve the existing vector store using its ID.
                return client.beta.vector_stores.retrieve(vector_store_id=db.id)
            
            
def create_index_if_not_exists(client, index_name):
    """Creates an index with the specified name if it does not already exist.
    
    Args:
        index_name (str): The name of the index to create or check for existence.
    
    Returns:
        Index: Instance of the created or existing index.
    """

    # List the names of existing indexes
    existing_indexes = [index_info['name'] for index_info in client.list_indexes()]

    # Check if the specified index name is not in the list of existing indexes
    if index_name not in existing_indexes:
        # Create the index with the specified parameters
        client.create_index(
            name=index_name,  # Name of the index
            dimension=3072,    # Dimension of the vectors
            metric='dotproduct',   # Similarity metric
            spec=ServerlessSpec(cloud='aws', region='us-east-1'), # Server configuration
        )

        # Wait until the index is ready to use
        while not client.describe_index(index_name).status['ready']:
            time.sleep(1)  # Pause to avoid overly frequent checks

        # Confirm that the index has been created
        print(f'Index {index_name} created')
    else:
        # Indicate that the index already exists
        print(f'The index {index_name} already exists')

    # Get an instance of the (created or existing) index and return it
    index = client.Index(index_name)
    return index


def upsert_vectors_in_batches(vectors, index, batch_size=100):
    """
    Upserts a list of vectors in batches to the provided index.

    This function splits the provided list of vectors into smaller batches and
    performs the upsert operation on each batch to avoid handling too much data at once.

    Args:
        vectors (list): A pre-generated list of vectors to upsert.
        index: An index instance with an upsert method to insert the vectors.
        batch_size (int): The number of vectors to include in each batch. Defaults to 100.
    
    Returns:
        None
    """
    # Process the list of vectors in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
         

def extract_pdf_text(pdf_path):
    """
    Extracts and structures the text from a PDF file using PyMuPDF (fitz). 

    Parameters:
    pdf_path (str): The path to the PDF.

    Returns:
    str: A string with all the PDF text.
    """
    
    # Open the PDF and extract the text
    doc = fitz.open(pdf_path)
    full_text = ''
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text('text')

    return full_text

     
def parsear_xml(xml_path):
    """
    Parsea un archivo XML y extrae el contenido de los párrafos dentro de la etiqueta <texto>.

    Args:
        xml_path (str): Ruta del archivo XML a procesar.

    Returns:
        str: Texto concatenado de todos los elementos <p> dentro de <texto>, separados por saltos de línea.
             Si la etiqueta <texto> no está presente, retorna una cadena vacía.
    """
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Buscar la etiqueta <texto>
    texto_element = root.find('texto')

    # Concatenar todo el contenido de los párrafos dentro de <texto>
    if texto_element is not None:
        xml_text = '\n'.join(
            elem.text.strip() for elem in texto_element.findall('.//p') if elem.text and elem.text.strip()
        )
    else:
        xml_text = ''

    return xml_text


def estructurar_boe(pdf_path, pdf_name):
    """
    Structures the text of the Spanish Constitution into titles, chapters, sections, and articles 
    using the position of the text and font size for improved categorization.

    Parameters:
    - pdf_path (str): Path to the Constitution PDF file.
    - pdf_name (str): The name of the Constitution PDF.

    Returns:
    - dict: A nested dictionary representing the hierarchical structure of the Constitution, 
            where each title contains chapters, chapters contain sections, and sections contain articles.
    """
    
    # Define patterns for different levels of the Constitution
    titulo_pattern = re.compile(r'(TÍTULO|TITULO)\s+\w+')
    capitulo_pattern = re.compile(r'(CAPÍTULO|CAPITULO)\s+\w+')
    seccion_pattern = re.compile(r'Sección\s+\w+')
    articulo_pattern = re.compile(r'(Artículo|Disposición|ANEXO)\s+(\d+|[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]+)')
    anexo_pattern = re.compile(r'ANEXO')
    anexo_chunks_pattern = re.compile(r'(Contenido ANEXO.*?)(?=\sCapítulo|\sSección|$)|' r'(Capítulo.*?)(?=\sSección|$)|' r'(Sección.*?)(?=$)', re.MULTILINE)
    indice_end_pattern = re.compile(r'TEXTO CONSOLIDADO')
    disposicion_pattern = re.compile(r'(DISPOSICIÓN|DISPOSICION|DISPOSICIONES)')
    boe_expressions_pattern = re.compile(r'Este texto consolidado no tiene valor jurídico.')

    estructura = {}
    estructura_final = {}
    current_titulo = 'TÍTULO SIN ASIGNAR'
    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
    current_seccion = 'Sección SIN ASIGNAR'
    current_articulo = 'Preámbulo'
    indice_flag = True
    end_flag = False
    disposicion_flag = False
    disposicion_name = None

    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                x0, y0, x1, y1 = element.bbox

                # Skip empty lines
                # Skip header lines or page markers
                # Skip BOE expressions
                if not text or y0 > 780 or y0 < 29 or re.match(boe_expressions_pattern, text):
                    continue

                # Detect the end of the table of contents 
                if indice_flag and re.match(indice_end_pattern, text):
                    indice_flag = False
                # Skip lines that belong to the table of contents
                if indice_flag or re.match(indice_end_pattern, text):
                    continue

                

                # Check if we are in disposiciones
                if re.match(disposicion_pattern, text):
                    font_names_disposicion = set()
                    for text_line in element:
                        for character in text_line:
                            # Revisamos si es un caracter y obtenemos su fuente
                            if isinstance(character, LTChar):
                                font_names_disposicion.add(character.fontname)
                    if any(negrita in name for name in font_names_disposicion for negrita in ["Bold", "Black"]):
                        disposicion_flag = True
                        disposicion_name = text
                        current_titulo = text
                        current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                        current_seccion = 'Sección SIN ASIGNAR'
                        current_articulo = str(disposicion_name + ' SIN ASIGNAR')
                    else:
                        pass
                # Detect the end of the disposiciones
                if disposicion_flag and (re.match(titulo_pattern, text) or re.match(capitulo_pattern, text) or re.match(seccion_pattern, text)):
                    disposicion_flag = False
                    disposicion_name = None

                # Identify Título
                if re.match(titulo_pattern, text) or re.match(anexo_pattern, text):
                    current_titulo = text
                    disposicion_flag = False
                    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                    current_seccion = 'Sección SIN ASIGNAR'
                    current_articulo = f'Contenido {text}' if re.match(anexo_pattern, text) else None

                # Identify Capítulo
                elif re.match(capitulo_pattern, text):
                    current_capitulo = text
                    current_seccion = 'Sección SIN ASIGNAR'
                    if current_articulo is not None:
                        match = re.search(anexo_chunks_pattern, current_articulo)
                        if match:
                            current_articulo = f'{match.group(1)} {current_capitulo}'

                # Identify Sección
                elif re.match(seccion_pattern, text):
                    current_seccion = text
                    if current_articulo is not None:
                        match = re.search(anexo_chunks_pattern, current_articulo)
                        if match:
                            elementos_filtrados = [e for e in [match.group(1), match.group(2)] if e is not None]
                            current_articulo = " ".join(elementos_filtrados)
                            current_articulo = f'{match.group(1)} {current_seccion}'
                
                # Identify Artículo, Disposición, Anexo
                elif re.match(articulo_pattern, text) and x0 < 92:
                    font_names_articulo = set()
                    for text_line in element:
                        for character in text_line:
                            # Revisamos si es un caracter y obtenemos su fuente
                            if isinstance(character, LTChar):
                                font_names_articulo.add(character.fontname)
                    if any(negrita in name for name in font_names_articulo for negrita in ["Bold", "Black"]):
                        current_articulo = text.replace('.', '')
                    else:
                        pass
                    
                # Identify Disposición
                elif disposicion_flag and not re.match(disposicion_pattern, text):
                    font_names = set()
                    for text_line in element:
                        for character in text_line:
                            # Revisamos si es un caracter y obtenemos su fuente
                            if isinstance(character, LTChar):
                                font_names.add(character.fontname)

                    if any(negrita in name for name in font_names for negrita in ["Bold", "Black"]) and re.match(re.compile(r'^[A-Za-zÁÉÍÓÚáéíóúÜüÑñ].*'), text):
                        current_articulo = str(disposicion_name + ' ' + text.split('.')[0])
                    else:
                        estructura.setdefault(current_titulo, {}) \
                                    .setdefault(current_capitulo, {}) \
                                    .setdefault(current_seccion, {}) \
                                    .setdefault(current_articulo, []).append(text)

                # Add content to the current Artículo
                elif current_articulo and not re.match(disposicion_pattern, text):
                    estructura.setdefault(current_titulo, {}) \
                              .setdefault(current_capitulo, {}) \
                              .setdefault(current_seccion, {}) \
                              .setdefault(current_articulo, []).append(text)

    estructura_final = {pdf_name: estructura}

    return estructura_final


def estructurar_codigo(pdf_path, pdf_name):
    """
    Structures the text of the Spanish Civil Code into books, titles, chapters, sections, and articles,
    while considering the position and font size of the text for improved categorization.

    Parameters:
    - pdf_path (str): Path to the Civil Code PDF file.
    - pdf_name (str): The name of the Civil Code PDF.

    Returns:
    - dict: A nested dictionary representing the hierarchical structure of the Civil Code,
            where each book contains titles, titles contain chapters, chapters contain sections, and sections contain articles.
    """
    
    # Define patterns for different levels of the Civil Code
    libro_pattern = re.compile(r'(LIBRO\s+\w+)')
    titulo_pattern = re.compile(r'(TÍTULO\s+\w+)')
    capitulo_pattern = re.compile(r'(CAPÍTULO|CAPITULO)\s+\w+')
    seccion_pattern = re.compile(r'(Sección\s+\w+)')
    articulo_pattern = re.compile(r'Artículo(?:s)?\s+(\d+)(?:\s+(?:a|y)\s+(\d+))?')
    disposicion_articulo_pattern = re.compile(r'(Disposición)\s+(\d+|[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]+)')
    derogated_pattern = re.compile(r'\(?(Derogado|Derogados|Suprimido|Sin contenido)\)?')
    indice_end_pattern = re.compile(r'TEXTO CONSOLIDADO')
    text_end_pattern = re.compile(r'JUAN CARLOS R.')
    disposicion_pattern = re.compile(r'(DISPOSICIÓN|DISPOSICION|DISPOSICIONES)')
    boe_expressions_pattern = re.compile(r'Este texto consolidado no tiene valor jurídico.')

    estructura = {}
    estructura_final = {}
    current_libro = 'LIBRO SIN ASIGNAR'
    current_titulo = 'TÍTULO SIN ASIGNAR'
    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
    current_seccion = 'Sección SIN ASIGNAR'
    current_articulo = 'Preámbulo'
    indice_flag = True
    end_flag = False
    disposicion_flag = False
    disposicion_name = None
    derogated_articles = []

    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                x0, y0, x1, y1 = element.bbox

                # Skip empty lines
                # Skip header lines or page markers
                # Skip BOE expressions
                if not text or y0 > 780 or y0 < 28 or re.match(boe_expressions_pattern, text):
                    continue

                # Detect the end of the table of contents 
                if indice_flag and re.match(indice_end_pattern, text):
                    indice_flag = False
                # Skip lines that belong to the table of contents
                if indice_flag or re.match(indice_end_pattern, text):
                    continue

                # Detect the end of the text
                if re.match(text_end_pattern, text):
                    end_flag = True
                # Skip lines that not belong to the main text
                if end_flag:
                    continue

                # Check if we are in disposiciones
                if re.match(disposicion_pattern, text):
                    disposicion_flag = True
                    disposicion_name = text
                    current_libro = text
                    current_titulo = 'TÍTULO SIN ASIGNAR'
                    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                    current_seccion = 'Sección SIN ASIGNAR'
                    current_articulo = str(disposicion_name + ' SIN ASIGNAR')
                # Detect the end of the disposiciones
                if disposicion_flag and (re.match(titulo_pattern, text) or re.match(capitulo_pattern, text) or re.match(seccion_pattern, text)):
                    disposicion_flag = False
                    disposicion_name = None

                # Identify Libro
                if re.match(libro_pattern, text):
                    current_libro = text
                    current_titulo = 'TÍTULO SIN ASIGNAR'
                    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                    current_seccion = 'Sección SIN ASIGNAR'
                    #current_articulo = None

                # Identify Titulo
                elif re.match(titulo_pattern, text):
                    current_titulo = text
                    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                    current_seccion = 'Sección SIN ASIGNAR'
                    #current_articulo = None

                # Identify Capitulo
                elif re.match(capitulo_pattern, text):
                    current_capitulo = text
                    current_seccion = 'Sección SIN ASIGNAR'
                    #current_articulo = None

                # Identify Seccion
                elif re.match(seccion_pattern, text):
                    current_seccion = text
                    #current_articulo = None

                # Identify Articulo
                elif (articulo_match := re.match(articulo_pattern, text)):
                    article_start = int(articulo_match.group(1))
                    article_end = int(articulo_match.group(2)) if articulo_match.group(2) else article_start
                    
                    for article_num in range(article_start, article_end + 1):
                        current_articulo = f'Artículo {article_num}'

                    # If 'Derogado' appears immediately after this line, mark articles as derogated
                    derogated_articles = [f'Artículo {i}' for i in range(article_start, article_end + 1)]
                
                # Check if the current line indicates derogation of the last detected articles
                elif derogated_articles and re.match(derogated_pattern, text):
                    for article in derogated_articles:
                        estructura.setdefault(current_libro, {}) \
                                  .setdefault(current_titulo, {}) \
                                  .setdefault(current_capitulo, {}) \
                                  .setdefault(current_seccion, {}) \
                                  .setdefault(article, []).append('Derogado')
                        #estructura[current_libro][current_titulo][current_capitulo][current_seccion][article] = ['Derogado']
                    derogated_articles = []

                # Identify Disposición
                elif disposicion_flag and not re.match(disposicion_pattern, text):
                    font_names = set()
                    for text_line in element:
                        for character in text_line:
                            # Revisamos si es un caracter y obtenemos su fuente
                            if isinstance(character, LTChar):
                                font_names.add(character.fontname)

                    if any(negrita in name for name in font_names for negrita in ["Bold", "Black"]) and re.match(re.compile(r'^[A-Za-zÁÉÍÓÚáéíóúÜüÑñ].*'), text):
                        current_articulo = str(disposicion_name + ' ' + text.split('.')[0])
                    else:
                        estructura.setdefault(current_libro, {}) \
                                    .setdefault(current_titulo, {}) \
                                    .setdefault(current_capitulo, {}) \
                                    .setdefault(current_seccion, {}) \
                                    .setdefault(current_articulo, []).append(text)
                    
                # Identify Disposición article
                elif re.match(disposicion_articulo_pattern, text) and x0 < 92:
                    current_libro = 'DISPOSICIONES'
                    current_titulo = 'TÍTULO SIN ASIGNAR'
                    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
                    current_seccion = 'Sección SIN ASIGNAR'
                    current_articulo = text.split('.')[0]

                # Add content to the current article
                elif current_articulo and not re.match(disposicion_pattern, text):
                    estructura.setdefault(current_libro, {}) \
                                  .setdefault(current_titulo, {}) \
                                  .setdefault(current_capitulo, {}) \
                                  .setdefault(current_seccion, {}) \
                                  .setdefault(current_articulo, []).append(text)

    estructura_final = {pdf_name: estructura}    

    return estructura_final


def estructurar_other_boe(pdf_path, pdf_name):
    """
    Structures the text of the Spanish Constitution into titles, chapters, sections, and articles 
    using the position of the text and font size for improved categorization.

    Parameters:
    - pdf_path (str): Path to the Constitution PDF file.
    - pdf_name (str): The name of the Constitution PDF.

    Returns:
    - dict: A nested dictionary representing the hierarchical structure of the Constitution, 
            where each title contains chapters, chapters contain sections, and sections contain articles.
    """
    
    # Define patterns for different levels of the Constitution
    capitulo_pattern = re.compile(r'(CAPÍTULO|CAPITULO)\s+\w+')
    seccion_pattern = re.compile(r'(Sección\s+\w+)')
    titulo_pattern = re.compile(r'(Título\s+\w+)')
    articulo_pattern = re.compile(r'(Artículo\s+(?:\d+|[a-záéíóúñü\s]+))')
    indice_end_pattern = re.compile(r'JEFATURA DEL ESTADO')
    pagination_pattern = re.compile(r'BOLETÍN OFICIAL DEL ESTADO|LEGISLACIÓN CONSOLIDADA|Página \d+')
    
    estructura = {}
    estructura_final = {}
    current_titulo = 'Título SIN ASIGNAR'
    current_capitulo = 'CAPÍTULO SIN ASIGNAR'
    current_seccion = 'Sección SIN ASIGNAR'
    current_articulo = 'Preámbulo'
    indice_flag = True

    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                x0, y0, x1, y1 = element.bbox

                # Skip empty lines
                # Skip header lines or page markers
                if not text or y0 > 780 or y0 < 29:
                    continue

                # Detect the start of the BOE content
                if indice_flag and re.match(indice_end_pattern, text):
                    indice_flag = False
                # Skip lines that not belong to the BOE content
                if indice_flag or re.match(indice_end_pattern, text):
                    continue

                # Identify Chapter
                if re.match(capitulo_pattern, text):
                    current_capitulo = text
                    estructura.setdefault(current_capitulo, {})
                    current_seccion = 'Sección SIN ASIGNAR'
                    current_titulo = 'Título SIN ASIGNAR'
                    current_articulo = None

                # Identify Section
                elif re.match(seccion_pattern, text):
                    current_seccion = text
                    estructura.setdefault(current_capitulo, {}).setdefault(current_seccion, {})
                    current_titulo = 'Título SIN ASIGNAR'
                    #current_articulo = None

                # Identify Title
                elif re.match(titulo_pattern, text):
                    current_titulo = text
                    estructura.setdefault(current_capitulo, {}).setdefault(current_seccion, {}).setdefault(current_titulo, {})
                    #current_articulo = None

                # Identify Article
                elif re.match(articulo_pattern, text):
                    current_articulo = text.replace('.', '')
                    estructura.setdefault(current_capitulo, {}) \
                              .setdefault(current_seccion, {}) \
                              .setdefault(current_titulo, {}) \
                              .setdefault(current_articulo, [])

                # Add content to the current article
                elif current_articulo:
                    estructura.setdefault(current_capitulo, {}) \
                              .setdefault(current_seccion, {}) \
                              .setdefault(current_titulo, {}) \
                              .setdefault(current_articulo, []).append(text)


    estructura_final = {pdf_name: estructura}

    return estructura_final



def ocr_imagen(imagen_bytes):
    """Applies OCR to the extracted image."""
    imagen = Image.open(io.BytesIO(imagen_bytes))
    texto_ocr = pytesseract.image_to_string(imagen)
    return texto_ocr


def extraer_imagenes_pdf(pdf_path):
    """
    Extracts images from the PDF using PyMuPDF (fitz).
    Returns a list of images in byte format.
    """
    imagenes = []
    documento = fitz.open(pdf_path)
    
    for pagina_num in range(len(documento)):
        pagina = documento[pagina_num]
        imagenes_pagina = pagina.get_images(full=True)
        
        for img_index, img in enumerate(imagenes_pagina):
            xref = img[0]  # Image reference
            base_imagen = documento.extract_image(xref)
            imagen_bytes = base_imagen['image']  # Extract the image in byte format
            imagenes.append(imagen_bytes)
    
    return imagenes


def estructurar_pdf_temario(pdf_path, pdf_name):
    """
    Structures the content of a syllabus PDF into a hierarchical format, including sections, subsections, 
    and sub-subsections. The organization is based on numerical patterns and element coordinates, 
    along with font size analysis.

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - pdf_name (str): Name of the PDF to encapsulate the structure in the result.

    Returns:
    - dict: Hierarchical structure of the syllabus, containing sections, subsections, and sub-subsections, 
            with each level including the corresponding text and additional content like OCR results 
            from images, if available.
    """
    
    # Define patterns for different levels in the Izeta PDFs
    number_pattern = re.compile(r'^\d+')
    section_pattern = re.compile(r'^\d+\.?(?!\d|\.) ?')
    subsection_pattern = re.compile(r'^\d+\.\d+\.?(?!\d|\.) ?')
    subsubsection_pattern = re.compile(r'^\d+\.\d+\.\d+ ?')
    introduccion_pattern = re.compile(r'INTRODUCCIÓN.')

    estructura = {}
    estructura_final = {}
    current_section = 'Sin Seccion'
    current_subsection = 'Sin Subseccion' 
    current_subsubsection = 'Sin Subsubseccion' 
    indice_flag = True  # Flag to determine if we are inside the table of contents
    list_indice_titulos = []

    # Extract images from the PDF
    # imagenes = extraer_imagenes_pdf(pdf_path)
    
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                x0, y0, x1, y1 = element.bbox

                # Skip lines that belong to the headline
                if (y0 > 790):
                    continue

                # Create a set with the font size of eache element text
                font_sizes = set()

                for text_line in element:
                    if isinstance(text_line, LTTextContainer):
                        for character in text_line:
                            if isinstance(character, LTChar):
                                font_sizes.add(character.size)

                # Skip lines with small font size
                if not font_sizes or (max(list(font_sizes)) < 8.1):
                    continue

                # Detect the end of the table of contents
                if indice_flag and re.match(section_pattern, text):
                    line_indice = text.split('.')[0]
                    if line_indice not in list_indice_titulos:
                        list_indice_titulos.append(line_indice)
                    else:
                        indice_flag = False

                # Detect the end of the table of contents
                if re.match(introduccion_pattern, text):
                    indice_flag = False


                # Skip lines that belong to the table of contents
                if indice_flag:
                    continue
                
                # Filter only text that corresponds to numbered sections
                if re.match(number_pattern, text) and (x0 < 72) and (max(list(font_sizes)) > 11.9):
                    if (max(list(font_sizes)) > 13.9) and re.match(section_pattern, text):  # Main section
                        current_section = text
                        current_subsection = 'Sin Subseccion'
                        current_subsubsection = 'Sin Subsubseccion' 
                        previous_text_section_flag = True
                    elif re.match(subsection_pattern, text):  # Subsection
                        current_subsection = text
                        current_subsubsection = 'Sin Subsubseccion' 
                        previous_text_section_flag = False
                    elif re.match(subsubsection_pattern, text):  # Sub-subsection
                        current_subsubsection = text
                        previous_text_section_flag = False

                # If it's not a new section, subsection, or sub-subsection, save the content
                elif text:
                    if re.match(number_pattern, text) and (y0 < 30):
                        continue
                    elif (max(list(font_sizes)) < 14.1) and (max(list(font_sizes)) > 13.9) and not re.match(number_pattern, text) and previous_text_section_flag:
                        # Complete section name between two pages
                        estructura.pop(current_section, None)
                        current_section += ''.join(text)
                        estructura[current_section] = {}
                        previous_text_section_flag = True
                        continue
                    else:
                        previous_text_section_flag = False
                    
                    estructura.setdefault(current_section, {}) \
                              .setdefault(current_subsection, {}) \
                              .setdefault(current_subsubsection, {}) \
                              .setdefault('content', []).append(text)

    estructura_final = {pdf_name: estructura}

    return estructura_final


def estructurar_pdf_temario_no_index(pdf_path, pdf_name):
    """
    Structures the content of a syllabus PDF into a hierarchical format, including sections, subsections, 
    and sub-subsections. The organization is based on numerical patterns and element coordinates, 
    along with font size analysis.

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - pdf_name (str): Name of the PDF to encapsulate the structure in the result.

    Returns:
    - dict: Hierarchical structure of the syllabus, containing sections, subsections, and sub-subsections, 
            with each level including the corresponding text and additional content like OCR results 
            from images, if available.
    """
    
    # Define patterns for different levels in the Izeta PDFs
    number_pattern = re.compile(r'^\d+')
    section_pattern = re.compile(r'^\d+\.(?!\d|\.) ?')
    subsection_pattern = re.compile(r'^\d+\.\d+\.?(?!\d|\.) ?')
    subsubsection_pattern = re.compile(r'^\d+\.\d+\.\d+ ?')
    titulo_pattern = re.compile(r'^TEMA.*')

    estructura = {}
    estructura_final = {}
    current_section = 'Sin Seccion'
    current_subsection = 'Sin Subseccion' 
    current_subsubsection = 'Sin Subsubseccion' 

    # Extract images from the PDF
    # imagenes = extraer_imagenes_pdf(pdf_path)
    
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                x0, y0, x1, y1 = element.bbox

                # Skip lines that belong to the headline
                if (y0 > 790):
                    continue

                # Create a set with the font size of eache element text
                font_sizes = set()

                for text_line in element:
                    if isinstance(text_line, LTTextContainer):
                        for character in text_line:
                            if isinstance(character, LTChar):
                                font_sizes.add(character.size)

                # Skip lines with small font size
                if not font_sizes or (max(list(font_sizes)) < 8.1) or re.match(titulo_pattern, text):
                    continue

                # Filter only text that corresponds to numbered sections
                if re.match(number_pattern, text) and (x0 < 72) and (max(list(font_sizes)) > 11.9):
                    if (max(list(font_sizes)) > 13.9) and re.match(section_pattern, text):  # Main section
                        current_section = text
                        current_subsection = 'Sin Subseccion'
                        current_subsubsection = 'Sin Subsubseccion' 
                        previous_text_section_flag = True
                    elif re.match(subsection_pattern, text):  # Subsection
                        current_subsection = text
                        current_subsubsection = 'Sin Subsubseccion' 
                        previous_text_section_flag = False
                    elif re.match(subsubsection_pattern, text):  # Sub-subsection
                        current_subsubsection = text
                        previous_text_section_flag = False

                # If it's not a new section, subsection, or sub-subsection, save the content
                elif text:
                    if re.match(number_pattern, text) and (y0 < 30):
                        continue
                    elif (max(list(font_sizes)) < 14.1) and (max(list(font_sizes)) > 13.9) and not re.match(number_pattern, text) and previous_text_section_flag:
                        # Complete section name between two pages
                        estructura.pop(current_section, None)
                        current_section += ''.join(text)
                        estructura[current_section] = {}
                        previous_text_section_flag = True
                        continue
                    else:
                        previous_text_section_flag = False
                    
                    estructura.setdefault(current_section, {}) \
                              .setdefault(current_subsection, {}) \
                              .setdefault(current_subsubsection, {}) \
                              .setdefault('content', []).append(text)

    estructura_final = {pdf_name: estructura}

    return estructura_final


def leer_archivo_como_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texto = file.read()
    return texto


def crear_prefijo(documento, pdf_name, libro=None, titulo=None, capitulo=None, seccion=None):
    """
    Creates a prefix string based on the document name and specified hierarchy levels (title, chapter, section).
    Additionally, it filters the document to ensure only relevant sections are processed.

    Parameters:
    -----------
    documento : dict
        The structured document from which to create the prefix.
    
    titulo : str, optional
        The title level within the document's hierarchy to include in the prefix (e.g., 'TÍTULO I'). Default is None.
    
    capitulo : str, optional
        The chapter level within the document's hierarchy to include in the prefix (e.g., 'CAPÍTULO PRIMERO'). Default is None.
    
    seccion : str, optional
        The section level within the document's hierarchy to include in the prefix (e.g., 'Sección 1'). Default is None.

    Returns:
    --------
    str
        A string representing the constructed prefix, including the document name and any specified hierarchy levels.
    dict
        The filtered portion of the document that corresponds to the specified hierarchy levels.
    """

    # Initialize the prefix as the pdf name
    prefijo = pdf_name
    documento = documento.get(pdf_name, {})

    # Build the prefix and filter the document based on the specified levels
    if libro:
        prefijo += f' \n {libro}'
        documento = documento.get(libro, {})
    if titulo:
        prefijo += f' \n {titulo}'
        documento = documento.get(titulo, {})
    if capitulo:
        prefijo += f' \n {capitulo}'
        documento = documento.get(capitulo, {})
    if seccion:
        prefijo += f' \n {seccion}'
        documento = documento.get(seccion, {})

    return prefijo, documento


def obtener_articulos_diccionario(documento, prefijo='', articulos=None, temario=False):
    """
    Extracts and concatenates articles from a structured legal document using the provided prefix. 
    Optionally filters for specific articles or processes a temario (table of contents).

    Parameters:
    -----------
    documento : dict
        The filtered portion of the document from which to extract articles.
    
    prefijo : str
        The prefix to be used in the article output.
    
    articulos : list of str, optional
        A list of specific articles to filter by. If not provided, all articles are processed.

    temario : bool, optional
        If True, process the document as a table of contents instead of full articles.

    Returns:
    --------
    dict
        A dictionary where each key is an article name, and the value is the longest string 
        (based on character count) associated with that article.
    """
    resultado = {}

    # If specific articles are provided, extract only those
    if articulos:
        for clave, valor in documento.items():
            if isinstance(valor, dict):
                nuevo_prefijo = f'{prefijo} \n {clave}' if prefijo else clave
                sub_resultado = obtener_articulos_diccionario(documento=valor, prefijo=nuevo_prefijo, articulos=articulos)
                for sub_clave, sub_valor in sub_resultado.items():
                    if sub_clave not in resultado:
                        resultado[sub_clave] = []
                    resultado[sub_clave].extend(sub_valor)
            elif clave in articulos:
                contenido_unido = ' '.join(valor).replace('\n', ' ').strip()
                clave_completa = f'{prefijo} \n {clave}'
                if clave not in resultado:
                    resultado[clave] = []
                resultado[clave].append(f'{clave_completa}: {contenido_unido}')
    else:
        if temario:
            # Process the document as a table of contents
            for clave, valor in documento.items():
                if isinstance(valor, dict):
                    nuevo_prefijo = f'{prefijo} \n {clave}' if prefijo else clave
                    sub_resultado = obtener_articulos_diccionario(documento=valor, prefijo=nuevo_prefijo, temario=temario)
                    for sub_clave, sub_valor in sub_resultado.items():
                        if sub_clave not in resultado:
                            resultado[sub_clave] = []
                        resultado[sub_clave].extend(sub_valor)
                else:
                    contenido_unido = ' '.join(valor).replace('\n', ' ').strip()
                    clave_completa = f'{prefijo} \n {clave}'
                    if clave_completa not in resultado:
                        resultado[clave_completa] = []
                    resultado[clave_completa].append(contenido_unido)
        else:
            # Process all articles without filtering
            for clave, valor in documento.items():
                if isinstance(valor, dict):
                    nuevo_prefijo = f'{prefijo} \n {clave}' if prefijo else clave
                    sub_resultado = obtener_articulos_diccionario(documento=valor, prefijo=nuevo_prefijo)
                    for sub_clave, sub_valor in sub_resultado.items():
                        if sub_clave not in resultado:
                            resultado[sub_clave] = []
                        resultado[sub_clave].extend(sub_valor)
                else:
                    contenido_unido = ' '.join(valor).replace('\n', ' ').strip()
                    clave_completa = f'{prefijo} \n {clave}'
                    if clave not in resultado:
                        resultado[clave] = []
                    resultado[clave].append(f'{clave_completa}: {contenido_unido}')

    return resultado


def procesar_tema(referencias, constitucion_estructurada, codigo_civil_estructurado, dict_temas):
    """
    Process a set of references for each legal document (Constitution, Civil Code, and Topics).
    
    Args:
        referencias (list): List of reference strings for legal documents and topics.
        constitucion_estructurada (dict): Structured data for the Spanish Constitution.
        codigo_civil_estructurado (dict): Structured data for the Civil Code.
        dict_temas (dict): Dictionary containing thematic data.
    
    Returns:
        dict: Dictionary with processed information organized by 'Constitución Española', 'Código Civil', and 'Temario'.
    """
    info_ce = procesar_referencias_ce(referencias, constitucion_estructurada)
    info_cc = procesar_referencias_cc(referencias, codigo_civil_estructurado)
    # info_temas = procesar_referencias_temas(referencias, dict_temas)
    
    return {
        'Constitución Española': info_ce,
        'Código Civil': info_cc,
        # 'Temario': info_temas
    }


def procesar_referencias_ce(referencias, constitucion_estructurada):
    """
    Process references specifically for the Spanish Constitution.
    
    Args:
        referencias (list): List of reference strings.
        constitucion_estructurada (dict): Structured data for the Spanish Constitution.
    
    Returns:
        list: List of articles referenced from the Constitution.
    """
    list_articulos_ce = []
    
    for referencia in referencias:
        # Information corresponding to the Spanish Constitution
        if 'CE.pdf' in referencia:
            info_referencia = referencia.split('- ')[1]
            # The various breakdowns of information for the Constitution
            for info in info_referencia.split(','):
                # Extract different parts of the naming convention for each distinct reference
                titulo, capitulo, seccion, articulos = None, None, None, None
                for parte in info.strip().split(';'):
                    # Create a flag to skip incorrect types
                    pass_flag = False
                    # Extract Title if declared
                    if re.search('Título', parte, re.IGNORECASE):
                        titulo = parte.upper()
                    # Extract Chapter if declared
                    elif re.search('Capítulo', parte, re.IGNORECASE):
                        capitulo = parte.upper()
                    # Extract Section if declared
                    elif re.search('Sección', parte, re.IGNORECASE):
                        seccion = parte
                    # Extract articles if declared
                    elif re.search(r'^([1-9]\d{0,3}|[1-9]\d{0,3}-[1-9]\d{0,3})$', parte):
                        articulos = extraer_articulos(parte)
                    else:
                        # If there is information not matching any expected type, create a flag to skip it
                        pass_flag = True
                if pass_flag:
                    # Skip parts without any Article reference
                    continue
                if articulos:
                    # If Articles are specified, return them directly
                    list_articulos_ce.extend(articulos)
                else:
                    # If Articles are not specified, extract all that match the prefix
                    prefijo, subdocumento_constitucion = crear_prefijo(documento=constitucion_estructurada, pdf_name='constitucion_española', titulo=titulo, capitulo=capitulo, seccion=seccion)
                    articulos_ce = obtener_articulos_diccionario(subdocumento_constitucion, prefijo).keys()
                    list_articulos_ce.extend(articulos_ce)
    
    return list_articulos_ce


def procesar_referencias_cc(referencias, codigo_civil_estructurado):
    """
    Process references specifically for the Civil Code.
    
    Args:
        referencias (list): List of reference strings.
        codigo_civil_estructurado (dict): Structured data for the Civil Code.
    
    Returns:
        list: List of articles referenced from the Civil Code.
    """
    list_articulos_cc = []
    
    for referencia in referencias:
        # Information corresponding to the Civil Code
        if 'Código_Civil.pdf' in referencia:
            info_referencia = referencia.split('- ')[1]
            # The various breakdowns of information for the Civil Code
            for info in info_referencia.split(', '):
                # Extract different parts of the naming convention for each distinct reference
                libro, titulo, capitulo, seccion, articulos = 'LIBRO SIN ASIGNAR', None, None, None, None
                for parte in info.split(';'):
                    # Create a flag to skip incorrect types
                    pass_flag = False
                    # Extract Book if declared
                    if re.search('Libro', parte, re.IGNORECASE):
                        libro = parte.upper()
                    # Extract Title if declared
                    elif re.search('Título', parte, re.IGNORECASE):
                        titulo = parte.upper()
                    # Extract Chapter if declared
                    elif re.search('Capítulo', parte, re.IGNORECASE):
                        capitulo = parte.upper()
                    # Extract Section if declared
                    elif re.search('Sección', parte, re.IGNORECASE):
                        seccion = parte
                    # Extract articles if declared
                    elif re.search(r'^([1-9]\d{0,3}|[1-9]\d{0,3}-[1-9]\d{0,3})$', parte):
                        articulos = extraer_articulos(parte)
                    else:
                        # If there is information not matching any expected type, create a flag to skip it
                        pass_flag = True
                if pass_flag:
                    # Skip parts without any Article reference
                    continue
                if articulos:
                    # If Articles are specified, return them directly
                    list_articulos_cc.extend(articulos)
                else:
                    # If Articles are not specified, extract all that match the prefix
                    prefijo, subdocumento_cc = crear_prefijo(documento=codigo_civil_estructurado, pdf_name='codigo_civil', libro=libro, titulo=titulo, capitulo=capitulo, seccion=seccion)
                    articulos_cc = obtener_articulos_diccionario(subdocumento_cc, prefijo).keys()
                    list_articulos_cc.extend(articulos_cc)
    
    return list_articulos_cc


def procesar_referencias_temas(referencias, dict_temas):
    """
    Process topic references based on structured thematic data.
    
    Args:
        referencias (list): List of reference strings.
        dict_temas (dict): Dictionary containing thematic data.
    
    Returns:
        list: List of topic references.
    """
    list_temas = []

    for referencia in referencias:
        # Information corresponding to topics
        if ('.pdf') in referencia and re.search(r'TEMA \d+', referencia, re.IGNORECASE):
            list_temas.append(referencia)
    
    return list_temas


def extraer_articulos(parte):
    """
    Extract articles from a specified range or single article string.
    
    Args:
        parte (str): String specifying the article(s), either as a single number or a range.
    
    Returns:
        list: List of article strings, each prefixed with 'Artículo'.
    """
    if '-' in parte:
        # Separate the numbers in the range
        inicio, fin = map(int, parte.split('-'))
        # Create the list of articles
        return [f'Artículo {i}' for i in range(inicio, fin + 1)]
    else:
        numero = int(parte)
        # The list will contain a single article
        return [f'Artículo {numero}']
    

def determinar_funcion_estructuracion(file_path):
    """
    Determines the appropriate function to parse and structure the file based on its content and format.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - str: Name of the function to use for parsing.
    """

    # Check if the file is an XML
    if file_path.name.split('.')[-1].lower() == 'xml':
        return "estructurar_xml"

    # # Check if the file is a TXT
    # if file_path.name.split('.')[-1] == 'txt':
    #     return "estructurar_txt"
    
    # Check only for PDFs files
    if file_path.name.split('.')[-1].lower() != 'pdf':
        raise ValueError(f"Formato no compatible: {file_path.name}. Se esperaba un archivo PDF o XML.")

    # Variables to track detected elements
    libro_detectado = False
    titulo_detectado = False
    capitulo_detectado = False
    seccion_detectado = False
    articulo_detectado = False
    boe_encabezado = False
    other_boe_disposiciones = False
    other_boe_flag = False
    temario_flag = False
    temario_final_flag = False
    indice_flag = False
    list_indice_sections = []

    # Analyze the PDF using pdfminer
    try:
        for page_layout in extract_pages(file_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    x0, y0, x1, y1 = element.bbox

                    # Create a set with the font size of eache element text
                    font_sizes = set()

                    for text_line in element:
                        if isinstance(text_line, LTTextContainer):
                            for character in text_line:
                                if isinstance(character, LTChar):
                                    font_sizes.add(character.size)

                    # Skip lines with small font size
                    if not font_sizes or (max(list(font_sizes)) < 8.1):
                        continue

                    # Detect "Libro" to prioritize 'estructurar_codigo'
                    if re.match(r'LIBRO\s+\w+', text, re.IGNORECASE):
                        libro_detectado = True

                    # Detect BOE elements
                    if re.match(r'TÍTULO\s+\w+', text, re.IGNORECASE):
                        titulo_detectado = True
                    if re.match(r'CAPÍTULO\s+\w+', text, re.IGNORECASE):
                        capitulo_detectado = True
                    if re.match(r'Sección\s+\w+', text, re.IGNORECASE):
                        seccion_detectado = True
                    if re.match(r'Artículo\s+(\d+|[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]+)', text, re.IGNORECASE):
                        articulo_detectado = True

                    if re.match(r'BOLETÍN OFICIAL DEL ESTADO', text, re.IGNORECASE) and (y0 > 700):
                        boe_encabezado = True

                    if boe_encabezado and re.match(r'I\.\s+DISPOSICIONES GENERALES', text):
                        other_boe_disposiciones = True
                    if boe_encabezado and other_boe_disposiciones and re.match(r'JEFATURA DEL ESTADO', text):
                        other_boe_flag = True

                    # Detect "TEMA"
                    if re.match(r'^TEMA\s+\d+(\s+[-–]?\s+.+)?', text, re.IGNORECASE):
                        temario_flag = True

                    # Detect Temario Index
                    if temario_flag:
                        # Check for index structure
                        section_match = re.match(r'^\d+\.(?!\d|\.) ?', text)
                        if section_match and (max(list(font_sizes)) > 11.9):
                            section_number = text.split('.')[0]
                            temario_final_flag = True
                            #GB print(section_number)
                            if section_number not in list_indice_sections:
                                list_indice_sections.append(section_number)
                            else:
                                indice_flag = True
                                break

            # Stop processing pages if temario and index are detected
            if (libro_detectado and articulo_detectado and boe_encabezado) or (boe_encabezado and (titulo_detectado or capitulo_detectado or seccion_detectado or articulo_detectado)) or (other_boe_flag) or (temario_flag and indice_flag):
                break
            

    except PDFSyntaxError as e:
        raise ValueError(f"Error al procesar el PDF: {e}")

    # Temario PDF Structure
    if temario_final_flag:
        if indice_flag:
            return "estructurar_pdf_temario"
        else:
            return "estructurar_pdf_temario_no_index"

    # BOE PDF Structure
    if libro_detectado and boe_encabezado:
        return "estructurar_codigo"

    elif boe_encabezado and (titulo_detectado or capitulo_detectado or seccion_detectado or articulo_detectado):
        return "estructurar_boe"
    
    if other_boe_flag:
            return "estructurar_other_boe"

    # Default function for unclassified PDFs
    return "extraer_texto_dividir_chunks"


def limpiar_articulos(documento_estructurado_articulos):
    """
    Elimina artículos repetidos en un diccionario, conservando el más completo (de mayor longitud).

    Args:
        documento_estructurado_articulos (dict): Diccionario donde las claves representan secciones o identificadores 
                                                 y los valores son listas de posibles versiones de un mismo artículo.

    Returns:
        dict: Diccionario procesado donde cada clave conserva solo el artículo más largo (más completo). 
              Si no hay artículos válidos, se asigna una cadena vacía.
    """

    # Cuando haya Artículos repetidos nos quedamos con el más completo
    for clave in documento_estructurado_articulos:
        valores_limpios = [str(valor).strip() for valor in documento_estructurado_articulos[clave] if isinstance(valor, str)]
        if valores_limpios:
            documento_estructurado_articulos[clave] = max(valores_limpios, key=len)
        else:
            documento_estructurado_articulos[clave] = ''

    return documento_estructurado_articulos


def procesar_articulos(articulos, max_tokens, model, codigo_flag=False):
    """
    Procesa y divide artículos en chunks si superan el límite de tokens.
    """

    articulos_split = {}
    enc = tiktoken.encoding_for_model(model)

    if codigo_flag:
        for articulo in articulos:
            text = articulos[articulo][0].split(':')[1:]
            text_final = ':'.join(text)
            text_final = " ".join(text_final.split())
            structure = articulos[articulo][0].split(':')[0]

            enc = tiktoken.encoding_for_model(model)
            text_tokens = len(enc.encode(text_final))

            if text_tokens > max_tokens:
                text_chunks = chunks_text(text=text_final, max_tokens=max_tokens, model=model)
                for chunk, chunk_idx in zip(text_chunks, range(len(text_chunks))):
                    new_artículo_name = f'{articulo}.{chunk_idx+1}'
                    chunk_content = f'{structure}: {chunk}'
                    articulos_split[new_artículo_name] = chunk_content
            else:
                articulos_split[articulo] = articulos[articulo][0]

    else:
        for articulo in articulos:
            text = articulos[articulo].split(':')[1:]
            text_final = ':'.join(text)
            structure = articulos[articulo].split(':')[0]

            text_tokens = len(enc.encode(text_final))
            #GB print(text_final)

            if text_tokens > max_tokens:
                text_chunks = chunks_text(text=text_final, max_tokens=max_tokens, model=model)
                for chunk, chunk_idx in zip(text_chunks, range(len(text_chunks))):
                    new_artículo_name = f'{articulo}.{chunk_idx+1}'
                    chunk_content = f'{structure} \n {chunk}'
                    articulos_split[new_artículo_name] = chunk_content
            else:
                articulos_split[articulo] = articulos[articulo]
    
    return articulos_split


def generate_context(document, chunk, model):
        """
        Generate context for a specific chunk using the language model.
        """

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        prompt = ChatPromptTemplate.from_template("""
        Eres un asistente especializado en el análisis de documentos legales y jurídicos, especialmente en temas relacionados con la Constitución Española y el Código Civil, enfocado para el temario de una oposición a policía nacional. Tu tarea es proporcionar un breve y relevante contexto para una sección específica del temario, situándola dentro del contenido general del documento de estudio.

        Aquí tienes el documento de estudio:
        <document>
        {document}
        </document>

        Aquí tienes el extracto específico que queremos contextualizar dentro del temario completo:
        <chunk>
        {chunk}
        </chunk>

        Proporciona un contexto conciso (1-2 oraciones) (chunk_size del orden de) para este extracto, siguiendo estas pautas:
        1. Identifica el tema o concepto jurídico principal abordado (por ejemplo, derechos fundamentales, competencias del Estado, obligaciones civiles).
        2. Menciona cualquier periodo o marco legal relevante si aplica (por ejemplo, artículos específicos o capítulos dentro de la Constitución Española).
        3. Nota cómo esta información se relaciona con el marco general de responsabilidades y competencias de los cuerpos de seguridad del Estado.
        4. Incluye cualquier referencia clave a artículos o secciones que provean un contexto importante.
        5. Evita frases como "Este extracto trata sobre" o "Esta sección proporciona". En su lugar, establece directamente el contexto.

        Por favor, proporciona un contexto breve y directo para mejorar la recuperación de este fragmento dentro del temario. Responde solo con el contexto y nada más.

        Contexto:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = llm.invoke(messages)
        return response.content


def procesar_temario(temario, max_tokens, model):
    """
    Procesa el contenido de un temario en formato de texto, generando fragmentos (chunks) y su contexto asociado.

    Args:
        temario (str): Contenido del temario en formato de texto.
        max_tokens (int): Número máximo de tokens permitidos por fragmento.
        model (str): Nombre del modelo de lenguaje utilizado para codificar el texto y generar el contexto.

    Returns:
        dict: Un diccionario donde cada clave representa un fragmento del temario. Cada valor es otro diccionario
              que contiene:
              - 'chunk_content': El contenido del fragmento de texto.
              - 'context_content': El contexto generado para ese fragmento.
    """

    tema_text = ''

    for k,v in obtener_articulos_diccionario(documento=temario, prefijo='', articulos=None, temario=True).items():
        # All PDF text
        tema_text += ''.join(v)

    # Generate Context for all Chunks
    chunks = []
    context_chunks = []

    for k,v in obtener_articulos_diccionario(documento=temario, prefijo='', articulos=None, temario=True).items():
        enc = tiktoken.encoding_for_model(model)
        text_tokens = len(enc.encode(v[0]))

        if text_tokens < max_tokens:
            chunk_content = f'{k}: \n {v[0]}'
            chunks.append(chunk_content)
            context_content = generate_context(document=tema_text, chunk=v[0], model=model)
            context_chunks.append(context_content)
        else:
            text_chunks = chunks_text(text=v[0], max_tokens=max_tokens, model=model)
            for chunk, chunk_idx in zip(text_chunks, range(len(text_chunks))):
                chunk_content = f'{k} Parte {chunk_idx+1}: \n {chunk}'
                chunks.append(chunk_content)
                context_content = generate_context(document=tema_text, chunk=v[0], model=model)
                context_chunks.append(context_content)

    # Create a json file with each chunk and context content
    dict_tema_chunks = {}

    for i in range(len(chunks)):
        dict_tema_chunks[i] = {
            'chunk_content': chunks[i], 
            'context_content': context_chunks[i]
        }

    return dict_tema_chunks


def procesar_documento(path, funcion_a_usar, model):
    """Determina el proceso a aplicar según el tipo de documento."""
    if funcion_a_usar in ['estructurar_boe', 'estructurar_other_boe']:
        documento = estructurar_boe(path, path.name) if funcion_a_usar == 'estructurar_boe' else estructurar_other_boe(path, path.name)
        articulos = limpiar_articulos(obtener_articulos_diccionario(documento))
        articulos_split = procesar_articulos(articulos, max_tokens=1024, model=model)
        guardar_json(path, articulos_split, boe_suffix=True)
    
    elif funcion_a_usar == 'estructurar_codigo':
        documento = estructurar_codigo(path, path.name)
        articulos = obtener_articulos_diccionario(documento)
        articulos_split = procesar_articulos(articulos, max_tokens=1000, model=model, codigo_flag=True)
        guardar_json(path, articulos_split)
    
    elif funcion_a_usar == 'extraer_texto_dividir_chunks':
        pdf_text = extract_pdf_text(path)
        text_chunks = chunks_text(text=pdf_text, max_tokens=512, model=model)
        pdf_estructurado_split = {idx: f'{path.name} \n {chunk}' for idx, chunk in enumerate(text_chunks)}
        guardar_json(path, pdf_estructurado_split)
    
    elif funcion_a_usar == 'estructurar_xml':
        xml_text = parsear_xml(path)
        text_chunks = chunks_text(text=xml_text, max_tokens=909, model=model)
        xml_estructurado_split = {idx: f'{path.name} \n {chunk}' for idx, chunk in enumerate(text_chunks)}
        guardar_json(path, xml_estructurado_split)

    elif funcion_a_usar in ['estructurar_pdf_temario', 'estructurar_pdf_temario_no_index']:
        temario = estructurar_pdf_temario(path, path.name) if funcion_a_usar == 'estructurar_pdf_temario' else estructurar_pdf_temario_no_index(path, path.name)
        temario_split = procesar_temario(temario, max_tokens=1500, model=model)
        guardar_json(path, temario_split, temario=True)


def procesar_paths(paths_list, model):
    """Itera sobre la lista de paths y ejecuta la función de estructuración adecuada."""
    for path in paths_list:
        funcion_a_usar = determinar_funcion_estructuracion(path)
        print('Path name: ', path.name)
        print(f'Función a utilizar: {funcion_a_usar}')
        procesar_documento(path, funcion_a_usar, model)


def save_json(filename, data):
    """
    Guarda el contenido en un archivo JSON.
    
    Parámetros:
    filename (str): Ruta donde se guardará el archivo JSON.
    data (dict o list): Datos a guardar en formato JSON (puede ser un diccionario o una lista).
    
    Retorna:
    bool: True si el archivo se guardó con éxito, False en caso de error.
    """
    try:
        # Crear las carpetas necesarias si no existen
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as archivo:
            json.dump(data, archivo, indent=4, ensure_ascii=False)
        return True
    except TypeError:
        print('Error: Los datos no son serializables a JSON.')
        return False
    except Exception as e:
        print(f'Ocurrió un error inesperado: {e}')
        return False


def guardar_json(path, documento_estructurado_split, boe_suffix=False, temario=False):
    """
    Guarda un diccionario en un archivo JSON en la ubicación adecuada, 
    transformando la ruta del directorio si es necesario.

    Args:
        path (Path): Ruta base del archivo a procesar.
        documento_estructurado_split (dict): Diccionario con la información estructurada a guardar.
        boe_suffix (bool, optional): Si es True, añade el sufijo "_boe" al nombre del archivo antes de guardarlo. 
                                     Por defecto, es False.

    Returns:
        None: La función guarda el archivo JSON en el directorio correspondiente.
    """
    
    if boe_suffix:
        path = path.with_name(path.stem + '_boe' + path.suffix)
    path_start = Path(str(path.parent).replace('01_raw', '02_intermediate'))
    if temario:
        json_name = path.parent.name.lower().replace(' ', '_') + '_chunks_structured.json'
    else:
        json_name = path.name.replace(path.suffix, '.json')
    documento_path = path_start / json_name
    save_json(documento_path, documento_estructurado_split)


def chunks_text(text, max_tokens, model):
    '''Splits a given text into multiple chunks, ensuring that each chunk does not exceed the specified maximum token count.
    
    This function splits the text by words and creates chunks that adhere to the token limit, ensuring that no words are
    split across chunks. This is useful for cases where you need to process text data in parts due to token limitations.
    
    Args:
        text (str): The input text to be split into chunks.
        max_tokens (int): The maximum number of tokens allowed per chunk.
        model (str): The name of the language model used for token encoding.
        
    Returns:
        List[str]: A list of text chunks, each with a token count less than or equal to max_tokens.
    '''

    # Import the appropriate encoding from tiktoken for the specified model
    enc = tiktoken.encoding_for_model(model)

    # Calculate the number of tokens in the input text
    text_tokens = len(enc.encode(text))

    # Check if the text exceeds the max_tokens limit
    if text_tokens > max_tokens:
        # If so, create a RecursiveCharacterTextSplitter to split the text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=(max_tokens/10),
            length_function=lambda text: len(enc.encode(text))
        )
        text_chunks = splitter.split_text(text)
    else:
        # If the text is within the limit, return it as a single chunk
        text_chunks = [text]

    return text_chunks


def dataframe_to_text_format(df, filepath):
    """
    Converts specific columns of a DataFrame into a custom text format
    and writes it to a file.

    Parameters:
    df (polars.DataFrame): The input DataFrame containing data.
    filepath (str): The path of the file where the formatted text will be written.

    The function processes the columns 'pregunta', 'respuesta_a', 'respuesta_b',
    'respuesta_c', 'respuesta_correcta', and 'retroalimentacion' from the DataFrame.
    It creates a text format where each question and its corresponding answers are written 
    along with an indicator for the correct answer and feedback.
    """
    # Select only the necessary columns from the DataFrame.
    df = df.select(pl.col(['pregunta', 'respuesta_a',
                           'respuesta_b', 'respuesta_c',
                           'respuesta_correcta', 'retroalimentacion']))
    
    # Open the file for writing in text mode.
    with open(filepath, 'w') as file:
        # Iterate over rows in the DataFrame.
        for row in df.iter_rows(named=True):
            # Write the question to the file in the specified format.
            pregunta = f'"*";"{row["pregunta"]}";""\n'
            
            # Determine the correct answer and prepare suffixes.
            respuesta_correcta = row['respuesta_correcta']
            a_suffix = '""'
            b_suffix = '""'
            c_suffix = '""'
            
            # Assign 'x' to the suffix of the correct answer.
            if respuesta_correcta == 'a)':
                a_suffix = '"x"'
            elif respuesta_correcta == 'b)':
                b_suffix = '"x"'
            else:
                c_suffix = '"x"'
            
            # Format the answers and their correctness indicator.
            respuesta_a = f'"";"{row["respuesta_a"].replace("a) ", "")}";{a_suffix}\n'
            respuesta_b = f'"";"{row["respuesta_b"].replace("b) ", "")}";{b_suffix}\n'
            respuesta_c = f'"";"{row["respuesta_c"].replace("c) ", "")}";{c_suffix}\n'
            
            # Format the feedback line.
            retroalimentacion = f'"@";"{row["retroalimentacion"]}";""\n'
            
            # Write all parts to the file.
            file.write(pregunta)
            file.write(respuesta_a)
            file.write(respuesta_b)
            file.write(respuesta_c)
            file.write(retroalimentacion)