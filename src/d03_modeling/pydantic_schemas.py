from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Ley(BaseModel):
    ley: str = Field(title='Ley',
                    description='Ley donde esta estipulado el uso del documento o formulario. Ejemplos: rd 557/2011 o lo 2/2009')

class DocumentoInfo(BaseModel):
    codigo_documento: Literal['EX-10', 'EX-17', 'no tiene'] = Field(title='Código',
    description="""Código identificador del documento. Suele aparecer arriba del todo.
    No intentes utilizar tu conocimiento para sacar el código.
    
    En caso de no tener poner 'no tiene'""")
    leyes: List[Ley] = Field(title='Leyes',
description="""Listado de leyes en las que se sustenta el documento. Suele aparecer en un recuadro arriba del todo.
En caso de no tener poner un string vacio""")
    
    
    
class Instrucción(BaseModel):
    numero: int = Field(title='Número',
                    description='Número de la instrucción')
    instruccion: str = Field(title='Instrucción',
                    description="""Instrucción sobre como completar un formulario.
En el caso de que la instrucción sea muy larga intenta simplificarla **PERO** asegurate de no perder partes importantes y deja los links""")

class Instrucciones(BaseModel):

    Instrucciones: List[Instrucción] = Field(title='Instrucciones',
description="""Listado de instrucciones de como completar un documento o formulario.""")
    
    
    
class SeccionDeImputacion(BaseModel):
    numero: int = Field(
        title='Número',
        description='Número de la sección.'
    )
    nombre: str = Field(
        title='Nombre',
        description='Nombre de la sección.'
    )
    informacion: str = Field(
        title='Información',
        description="""Proporciona información sobre la secciçon que tipo de datos tiene que imputar.
Estos datos a imputar pueden ser introducidos de forma escrita o seleccionando la opción que corresponde (esta pre-definida) para estos casos **NO HACE FALTA** que expliques todas las opciones pero explica la sección a la que pertenecen.
En el caso de que hayan detalles y conceptos que un usuario migrante no pueda entender, como por ejemplo abreviaciones, diminutivos o referencias a otros documentos intenta explicarlos brevemente.""")

class Formulario(BaseModel):
    secciones: List[SeccionDeImputacion] = Field(
        title='Secciones de imputación',
        description='Secciones donde la mayoría de campos son a rellenar por el usuario, no de seleccionar opciones. Selecciona solo las secciones principales'
    )
    
class SiNo(BaseModel):
    resultado: Literal['si', 'no'] = Field(
        title='Resultado',
        description='Confirmación con un "si" o con un "no" si el documento que se te ha dispuesto se encuentra en el texto'
    )
    
    
# Pydantic class that only allows the specified rotation angles.
class RotationResponse(BaseModel):
    rotation: Literal['-90', '0', '90', '180'] = Field(
        title='Rotation',
        description=(
"""Rotation angle (in degrees) required to straighten the text in the image
Allowed values:
'90': rotate 90° counter-clockwise,
'-90': rotate 90° clockwise,
'180': rotate 180°,
'0': no rotation required."""
        )
    )
    
    
class Campo(BaseModel):
    nombre_campo: str = Field(title='Nombre', description='Nombre del campo a imputar. Ejemplos son ["Representante legal, en su caso", "Nacionalidad", "Nombre del padre", etc]')
    nombre_completo: str = Field(title='Nombre Completo', description="""En el caso de que sea un diminutivo, apreviación o siglas, pon su nombre completo.
Por ejemplo: D.N.I = documento nacional de identidad, NIE = documento de identidad de Extranjero, C.P = Código Postal, etc.
En en el caso de no tener o ser le mismo campo, dejarlo vacio""")

class Campos(BaseModel):
    campos: List[Campo] = Field(title='Leyes',
description="""Listado de campos a rellenar por el usuario de forma escrita""")
    

class Seleccion(BaseModel):
    nombre_campo: str = Field(title='Nombre', description='Nombre del campo a seleccionar')
    seleccionado: Literal['true', 'false'] = Field(title='Seleccionado?', description='true en el caso de que haya sido seleccionado por el usuario y false en el caso de que no')

class Selecciones(BaseModel):
    campos: List[Seleccion] = Field(title='Leyes',
description="""Listado de las distintas opciones a seleccionar por el usuario""")
    
    
class Traduccion(BaseModel):
    palabra: str = Field(title='Palabra', description='Palabra original que se va a traducir')
    traduccion:str = Field(title='Traduccion', description='Traduccion de la palabra al idioma elegido')

class Traducciones(BaseModel):
    traducciones: List[Traduccion] = Field(title='Traducciones',
description="""Listado de palabras con sus traducciones""")
    
    
class Resumen(BaseModel):
    resumen: str = Field(title='Resumen',
description="""Resumen sobre el formulario o documento""")
    
    
class TraduccionFormateada(BaseModel):
    traduccion_formateada: str = Field(title='Traduccion Formateada',
description="""Traducción dado un formato markdown adecuado a la información""")