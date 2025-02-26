EXTRACT_FORM_TYPE = """De la siguiente imagen de un documento de un proceso de inmigración española.
Extrae tanto el código del cocumento como las leyes en las que se sustenta.
"""

EXTRACT_FORM_INSTRUCTIONS = """De la siguiente imagen de un documento de un proceso de inmigración española.
Extrae las instrucciones a seguir para poder completarlo.
"""

EXTRACT_FORM_MAIN_SECTIONS = """De la siguiente imagene de un documento de un proceso de inmigración española.
Identifica todas las **SECCIONES PRINCIPALES** del documento para explicarselas al usuario."""


IS_DOC_IN_TEXT_PROMPT = """Analiza el siguiente texto y verifica si se menciona el documento especificado.
Responde únicamente con "sí" si el nombre del documento aparece en el texto o "no" si no se encuentra.
No añadas explicaciones ni comentarios adicionales."""

MAIN_SYSTEM_PROMPTIS_IN_TEXT = 'Eres un bot que tiene como finalidad encontrar si el nombre de un documento esta referido en un texto'

MAIN_SYSTEM_PROMPTIS_ROTATE  = """You are an expert image orientation analyst.
Analyze the provided image and determine the rotation needed so that all the text is horizontal.
The rotation is one of the following values:
'90' (rotate 90° counter-clockwise), '-90' (rotate 90° clockwise), '180', or '0'."""


MAIN_SYSTEM_PROMPT_DOCT_ID = 'Eres un analista de documentos sobre la inmigración española y tu objetivo es identificar el código del documento que estas procesando'


def is_doc_present_in_text(doc_name, text):
    return """{}
CÓDIGO DEL DOCUMENTO A BUSCAR: {}\n
Texto donde buscar el código del documento: {}""".format(IS_DOC_IN_TEXT_PROMPT, doc_name, text)


def is_doc_present_in_text_message(documents):
    return [
    [
        {
            'role': 'system',
            'content': MAIN_SYSTEM_PROMPTIS_IN_TEXT,
        },
        {
            'role': 'user',
            'content': is_doc_present_in_text(doc_name, text),
        }
    ]
    for doc_name, text in documents
]


MAIN_SYSTEM_PROMPT_INPUT_SECTIONS="""Tu objetivo es analizar formularios utilizados en distintos procesos de inmigración española.
Por cada documento, deberás extraer únicamente aquellos campos que los usuarios deben imputar."""

INPUT_SECTIONS_PROMPT= """A partir de la imagen de un formulario de un proceso de inmigración española, identifica y extrae exclusivamente los campos que el usuario debe rellenar de forma escrita.
**SOLO** incluye aquellos campos que requieren entrada manual, **IGNORA** los campos de selección múltiple o aquellos que impliquen marcar una opción.
**NO DUPLIQUES CAMPOS** solo han de aparecer una vez."""

SELECT_SECTIONS_PROMPT = """
A partir de la imagen de un formulario de un proceso de inmigración en España, identifica y extrae únicamente los campos de selección múltiple o aquellos que impliquen marcar una opción en algún recuadro.

**IMPORTANTE:**  
- IGNORA los campos que requieren entrada de texto por parte del usuario.  
- Ten en cuenta que pueden existir selecciones anidadas.  
  Por ejemplo, si existe un campo llamado "TIPO DE AUTORIZACIÓN SOLICITADA" con las opciones:
  [□ RESIDENCIA INICIAL, □ PRÓRROGA DE RESIDENCIA, □ RENOVACIÓN ESPECIAL DE RESIDENCIA, □ RESIDENCIA Y TRABAJO],
  estas pueden ser a su vez el nombre de otros campos con otras opciones a seleccionar.

**NO DUPLIQUES CAMPOS:**  
CADA NOMBRE DEL CAMPO SOLO PUEDE APARECER UNA VEZ, ELIMINA DUPLICADOS.
"""


def SYSTEM_PROMPT_TRANSLATE(language):
    
    if language == 'chino_mandarin':
        extra_chino = ' (Usa caracteres hanzi, no hace falta el uso del pinyin)'
    else:
        extra_chino = ''
    
    return f"""Eres un traductor profesional y experto en contextos culturales y lingüísticos.
A continuación, te proporcionaré una lista de palabras.
Tu tarea es traducir cada palabra de forma precisa y manteniendo el significado original al idioma {language}{extra_chino}.
Si alguna palabra tiene múltiples acepciones, elige la que resulte más adecuada al contexto general o, si es necesario, proporciona breves variantes explicativas"""


def TRANSLATE_PROMPT(language, palabras, contexto):
    
    return f"""Eres un traductor profesional con amplio conocimiento en matices culturales y contextuales. Se te proporcionará:
    - El idioma de destino: {language}
    - Un bloque de texto crudo que contiene el contexto donde se encuentran las palabras.
    - Una lista de frases o palabras a traducir. Cada línea es una frase o palabra, no mezcles ni combines palabras o frases tienen que traducirse por separado.
    - **NO TRADUZCAS** puntos consecutivos como "…………" NI SIGNOS O CARACTERES ESPECIALES COMO "□"
    
    Tu tarea es traducir cada palabra al idioma {language} de forma precisa. Utiliza el texto de contexto para elegir la traducción más adecuada en caso de ambigüedades. La respuesta deberá organizarse en una lista donde cada elemento incluya:
    - La palabra original (sin modificaciones).
    - La traducción correspondiente.

    Text en crudo de contexto:
    {contexto}

    Palabras o frases a traducir:
    {palabras}
    """
    
    
SYSTEM_PROMPT_RESUMEN = """
Dada la siguiente información sobre un documento, por favor elabora un resumen que incluya:

- La finalidad o el propósito principal del documento.
- Las características o cualidades más importantes que lo definen.
- El público o contexto para el cual fue creado.
- Cualquier otro aspecto relevante que destaque su utilidad y valor.

Utiliza un lenguaje claro y conciso, y asegúrate de cubrir todos los puntos importantes mencionados en la información proporcionada."""


def SYSTEM_PROMPT_FORMATED_TRANSLATION(language):
    
    return f"""Eres un traductor profesional con un profundo conocimiento de los contextos culturales y lingüísticos.
A continuación, recibirás un texto en español. Tu tarea es traducirlo al idioma {language}. Si el texto ya está en español, no es necesario realizar ninguna traducción adicional.
**Por favor**, utiliza **FORMATO MARDOWN** para presentar la información de manera clara, concisa y estructurada.
**EVITA PONER TÍTULOS, LA ETIQUETA MÁS GRANDE DEBERIAN DE SER SUBTITULOS** """


def create_formated_translation_message(documents):
    return [
    [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT_FORMATED_TRANSLATION(language),
        },
        {
            'role': 'user',
            'content': f'Aquí tienes el texto para traducir y dar formato mardown adecuado en el idioma {language}:\n{texto_a_traducir}',
        }
    ]
    for language, texto_a_traducir in documents
]