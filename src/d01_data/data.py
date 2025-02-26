from pinecone import ServerlessSpec
import time
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
import os
import unicodedata
import json

import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import cohere
from pdf2image import convert_from_path



load_dotenv(os.path.dirname(__file__)+"/../../.env")


def read_json(filename):
    """
    Lee un archivo JSON y devuelve su contenido.
    
    Parámetros:
    filename (str): Ruta del archivo JSON a leer.

    Retorna:
    dict o list: Contenido del archivo JSON como un diccionario o lista.
    """
    try:
        with open(filename, 'r', encoding="utf-8") as archivo:
            contenido = json.load(archivo)
        return contenido
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{filename}' no es un JSON válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

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
        
        
def save_text_to_file(text: str, file_path: str):
    """
    Saves a given text string to a specified file path as a .txt file.

    :param text: The text to save.
    :param file_path: The path where the file should be saved.
    """
    
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f'File saved successfully at: {file_path}')
    except Exception as e:
        print(f'Error saving file: {e}')

        
 
def create_index_if_not_exists(client, index_name):
    """Creates an index with the specified name if it does not already exist.
    
    Args:
        index_name (str): The name of the index to create or check for existence.
    
    Returns:
        Index: Instance of the created or existing index.
    """
    
    existing_indexes = [index_info['name'] for index_info in client.list_indexes()]
    if index_name not in existing_indexes:
        client.create_index(name=index_name, dimension=3072, metric='dotproduct', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        while not client.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f'Index {index_name} created')
    else:
        print(f'The index {index_name} already exists')

    index = client.Index(index_name)
    return index


def setup_db(index_name):
    pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])
    pc_index = create_index_if_not_exists(client=pc, index_name=index_name)
    return pc_index


def setup_embeddings():
    return OpenAIEmbeddings(model=st.secrets['EMBEDDING_MODEL'], dimensions=3072)


def setup_encoding():
    return tiktoken.encoding_for_model(st.secrets['OPENAI_MODEL'])


def setup_cohere():
    return cohere.ClientV2(api_key=st.secrets['COHERE_API_KEY'])


def setup_model():
    return  ChatOpenAI(
        model=st.secrets['OPENAI_MODEL'],
        api_key = st.secrets['OPENAI_API_KEY'],
        temperature=0.7,
        max_tokens=600,
        top_p=0.6,
        timeout=None,
        max_retries=2
    )


def setup_env(pc_index_name):
    pc_index = setup_db(pc_index_name)
    embeddings = setup_embeddings()
    encoding = setup_encoding()
    cohere = setup_cohere()
    model = setup_model()
    
    return pc_index, embeddings, encoding, cohere, model


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

        
def get_all_id_prefixes(index, level=1):
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
        prefixes = {'#'.join(_id.split('#')[:level]) for _id in ids}
        # Merge the temporary set with the main prefix set
        prefixes_set |= prefixes 

    # Print the number of unique prefixes found (in Spanish)
    # print(f'Se ha encontrado {len(prefixes_set)} en el índice')

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
         
         
         


def pdf_to_images(pdf_path, dpi=200, fmt="jpeg", output_dir=None):
    """
    Convert each page of a PDF into an image and save them in a folder named after the PDF.
    
    Parameters:
        pdf_path (str): The file path to the PDF.
        dpi (int): The resolution of the output images. Default is 200.
        fmt (str): The format of the output images ('jpeg', 'png', etc.). Default is 'jpeg'.
    """
    # Get the base name of the PDF without extension (e.g., 'document' from 'document.pdf')
    pdf_path = Path(pdf_path)
    base_name = pdf_path.stem
    
    if not output_dir:
        # Create an output folder in the same directory as the PDF, named after the PDF file
        output_dir = str(pdf_path.parent / base_name)
    else:
         output_dir = str(Path(output_dir) / base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF pages to images
    print(f"Converting '{pdf_path}' to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    images_names = []
    # Save each page as an image in the output folder
    for i, page in enumerate(pages, start=1):
        image_filename = f"{base_name}_page_{i}.{fmt}"
        image_path = os.path.join(output_dir, image_filename)
        images_names.append(images_names)
        page.save(image_path, fmt.upper())
        print(f"Saved page {i} as {image_filename}")
    
    print(f"Conversion complete! {len(pages)} images saved in '{output_dir}'.")

    return images_names



def get_orientacion_docs(orientacion_file, keys=['Documentación exigible', 'Procedimiento']):
    """
    Read and process orientation documentation from the given JSON file.

    Args:
        orientacion_file (str): The path to the JSON file containing orientation data.

    Returns:
        tuple: A tuple containing:
            - process_name (str): The name of the process as extracted from the first matching chunk.
            - data (str): A newline-joined string of all chunk contents that contain 'Documentación exigible'.
    """
    # Read JSON data from the file.
    raw_data = read_json(orientacion_file)
    data = []
    # Filter out and collect 'chunk_content' fields that contain 'Documentación exigible'.
    for key in keys:
        data += [v['chunk_content'] for _, v in raw_data.items() if key in v['chunk_content']]
    
    if len(data) > 0:
    
        # The process name is taken from the first line of the first matching chunk.
        process_name = data[0].split('\n')[0]
        
        # Join all matching chunk contents into a single multi-line string.
        data = '\n'.join(data)
        
        # Return the process name and the combined chunk contents.
        return process_name, data

    else:
        return None, None
