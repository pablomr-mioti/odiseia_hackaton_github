from pydantic import BaseModel, Field
from typing import Literal
from PIL import Image

from google.genai import types
from langchain_openai import ChatOpenAI

from src.d03_modeling.pydantic_schemas import (RotationResponse, DocumentoInfo,
                                               Campos, Selecciones,
                                               Traducciones, Resumen,
                                               TraduccionFormateada)
from src.d03_modeling.prompts import (MAIN_SYSTEM_PROMPTIS_ROTATE, EXTRACT_FORM_TYPE,
                                      MAIN_SYSTEM_PROMPTIS_IN_TEXT, MAIN_SYSTEM_PROMPT_INPUT_SECTIONS,
                                      INPUT_SECTIONS_PROMPT, SELECT_SECTIONS_PROMPT,
                                      SYSTEM_PROMPT_TRANSLATE, TRANSLATE_PROMPT,
                                      SYSTEM_PROMPT_RESUMEN, SYSTEM_PROMPT_FORMATED_TRANSLATION,
                                      create_formated_translation_message)

def get_image_rotation(client, model: str, image_path: str) -> int:
    """
    Given the path to an image, this function uses an LLM model to determine the rotation
    required so that the text appears horizontal. The returned rotation value will be one of the following:
    '90' (counter-clockwise), '-90' (clockwise), '180', or '0' (if no rotation is needed).
    """
    # Load the image using PIL.
    image = Image.open(image_path)

    try:
        # Configure the request for the LLM model.
        config = types.GenerateContentConfig(
            system_instruction=MAIN_SYSTEM_PROMPTIS_ROTATE,
            max_output_tokens=70,
            top_p=0.6,
            temperature=0.5,
            response_mime_type='application/json',
            response_schema=RotationResponse,
        )

        # Send the image to the model along with the instruction.
        response = client.models.generate_content(
            model=model,
            contents=['Determine the rotation needed to straighten the text in the image.', image],
            config=config,
        )

        # Extract the rotation from the parsed response.
        rotation = response.parsed.rotation
        rotation = int(rotation)
         
    except Exception as e:
        print(f'Error during rotation inference {str(e)}')
        print('Setting the rotation to 0')
        rotation = 0
        
    # Print the selected rotation to the console.
    print(f'Selected rotation: {rotation}º')

    # Return the rotation value.
    return rotation


def get_doc_id_from_image(client, model, image):
    
    try:
        config = types.GenerateContentConfig(
                    system_instruction=MAIN_SYSTEM_PROMPTIS_IN_TEXT,
                    max_output_tokens=150,
                    top_p= 0.6,
                    temperature= 0.5,
                    response_mime_type= 'application/json',
                    response_schema=DocumentoInfo,
                )
        
        doc_info = client.models.generate_content(
                    model=model,
                    contents=[EXTRACT_FORM_TYPE, image],
                    config=config)
        
        return doc_info.parsed
        
    except Exception as e:
        print(f'Error during doc id inference {str(e)}')
        return None



def get_imputation_campos(client, model, pdf):
    
    config = types.GenerateContentConfig(
                system_instruction=MAIN_SYSTEM_PROMPT_INPUT_SECTIONS,
                max_output_tokens=4_000,
                top_p= 0.7,
                temperature=0,
                response_mime_type= 'application/json',
                response_schema=Campos,
            )
    
    
    campos_a_imputar = client.models.generate_content(
            model=model,
                contents=[
                    types.Part.from_bytes(
                        data=pdf,
                        mime_type='application/pdf'),
                    INPUT_SECTIONS_PROMPT],
                config=config)
    
    
    return campos_a_imputar

def get_imputation_seleccion(client, model, pdf):
    
    config = types.GenerateContentConfig(
                system_instruction=MAIN_SYSTEM_PROMPT_INPUT_SECTIONS,
                max_output_tokens=4_000,
                top_p= 0.6,
                temperature=0,
                response_mime_type= 'application/json',
                response_schema=Selecciones,
            )
    
    campos_a_seleecionar = client.models.generate_content(
            model=model,
                contents=[
                    types.Part.from_bytes(
                        data=pdf,
                        mime_type='application/pdf'),
                    SELECT_SECTIONS_PROMPT],
                config=config)
    
    
    return campos_a_seleecionar



def translate(client, model, language, palabras, contexto):
    config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_TRANSLATE(language),
                max_output_tokens=6_000,
                top_p= 0.6,
                temperature=0.5,
                response_mime_type= 'application/json',
                response_schema=Traducciones,
            )
    
    traducciones = client.models.generate_content(
            model=model,
                contents=[TRANSLATE_PROMPT(language, palabras, contexto)],
                config=config)
    
    return traducciones


def create_resumen(client, model, resumen):
    config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_RESUMEN,
                max_output_tokens=500,
                top_p= 0.6,
                temperature=0,
                response_mime_type= 'application/json',
                response_schema=Resumen,
            )

    resumen = client.models.generate_content(
            model=model,
                contents=[f'Aquí tienes la información del documento a resumir:\n{resumen}'],
                config=config)
    
    return resumen


def gemini_translate_and_format(client, model, language, texto_a_traducir):
    config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_FORMATED_TRANSLATION(language),
                max_output_tokens=5_000,
                top_p=0.6,
                temperature=0.3,
                response_mime_type= 'application/json',
                response_schema=TraduccionFormateada,
            )

    traducciones = client.models.generate_content(
            model=model,
                contents=[f'Aquí tienes el texto a traducir en el idioma {language}:\n{texto_a_traducir}'],
                config=config)
    
    return traducciones



async def openai_translate_and_format(documents, model):

    messages = create_formated_translation_message(documents)
    
    llm = ChatOpenAI(
        model=model,
        temperature=0.6,
        max_tokens=4_000,
        top_p=0.6,
        timeout=None,
        max_retries=2
    ).with_structured_output(TraduccionFormateada, include_raw=False, method ='json_schema', strict=True)
    
    traducciones = await llm.abatch(messages)
    
    return traducciones