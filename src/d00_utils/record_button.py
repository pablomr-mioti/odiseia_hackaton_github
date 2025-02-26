

import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import numpy as np
import PyPDF2
import io
import base64
from pydub import AudioSegment


def extraer_texto_pdf(file):
    """Extrae el texto de un archivo PDF."""
    try:
        pdfReader = PyPDF2.PdfReader(file)
        texto = ""
        for page in pdfReader.pages:
            texto += page.extract_text() or ""
        return texto
    except Exception as e:
        st.error(f"Error al extraer texto del PDF: {e}")
        return ""

def image_to_base64(image_file):
    if image_file is not None:
        return base64.b64encode(image_file.read()).decode("utf-8")

idiomas = {
    "Español": "es-ES",
    "Inglés": "en-US",
    "Francés": "fr-FR",
    "Alemán": "de-DE",
    "Italiano": "it-IT",
    "Arabe": "ar-AR",
    "Chino": "zh-CN",
    "Japonés": "ja-JP",
    }

idioma_seleccionado = st.selectbox("Selecciona un idioma:", list(idiomas.keys()))

st.title("Grabación y Transcripción de Audio")

# st.markdown("""
# Este ejemplo usa el componente `streamlit-audiorec` para capturar audio desde el navegador.
# Luego utilizamos `SpeechRecognition` para transcribirlo.
# """)
#st.title("Interfaz de Chat por Voz con LLM")


# -------------------------------
# Sección: Subir archivo PDF
# -------------------------------
st.sidebar.header("Subir PDF")
pdf_file = st.sidebar.file_uploader("Elige un archivo PDF", type="pdf")
img_file = st.sidebar.file_uploader("Elige un archivo de imagen", type=["png", "jpg", "jpeg"])




audio_data = st_audiorec()

# 2. Procesar el audio cuando el usuario detiene la grabación
if audio_data is not None:

    audio_bytes = io.BytesIO(audio_data)

    audio_segment = AudioSegment.from_file(audio_bytes, format="wav")

    with io.BytesIO() as wav_buffer:
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_buffer) as source:
            audio_data_sr = recognizer.record(source)  # lee la totalidad del audio
        try:
            texto_transcrito = recognizer.recognize_google(audio_data_sr, language=idiomas[idioma_seleccionado])
            st.subheader("Texto transcrito:")
            st.write(texto_transcrito)
        except Exception as e:
            st.error(f"Error al transcribir: {e}")
    

    st.audio(audio_data, format="audio/wav")
else:
    st.info("Presiona 'Start Recording' para grabar, y luego 'Stop Recording' para transcribir.")



if pdf_file is not None:
    pdf_text = extraer_texto_pdf(pdf_file)
    st.sidebar.text_area("Contenido del PDF", pdf_text, height=300)
    
if  img_file is not None:
    img_64 = image_to_base64(img_file)
    st.sidebar.image(img_file, use_container_width=True)   
    
# -------------------------------
# Sección: Enviar al LLM
# -------------------------------
# if st.button("Enviar al LLM"):
#     if texto_transcrito.strip() == "":
#         st.error("No se pudo obtener texto de la grabación.")
#     else:
#         try:
#             st.subheader("Texto transcrito:")
#             st.subheader(texto_transcrito)
#         except Exception as e:
#             st.error(f"Error al comunicarse con el LLM: {e}")


