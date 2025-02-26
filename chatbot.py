import os
from pathlib import Path
import streamlit as st
import wave
import whisper
from gtts import gTTS
from langchain_core.messages import HumanMessage, SystemMessage

from src.d00_utils.utils import custom_css, get_image_base64, get_content_languages_dict
from src.d01_data.agent import RagAgent

image_path = os.path.dirname(os.path.abspath(__file__)) + '/src/d00_utils/images'

st.set_page_config(layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)
# Obtener la imagen en formato base64
image_base64 = get_image_base64(image_path+'/fondo-curvas.png')

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

archivo_wav = "grabacion.wav"
if Path(archivo_wav).exists():
    os.remove(archivo_wav)
samplerate = 44100  # Frecuencia de muestreo (Hz)

languages_dict = {
    "🇬🇧 English": "en",
    "🇪🇸 Español": "es",
    "🇫🇷 Français": "fr",
    "🇩🇪 Deutsch": "de",
    "🇸🇦 العربية": "ar",
    "🇨🇳 中文": "zh",
    "🇺🇦 Українська": "uk"
}

content_languages_dict = get_content_languages_dict()

def get_audio():
    wav_audio_data = st.audio_input(content_languages_dict[st.session_state.language_selected]["Record a voice message"])
    if wav_audio_data is not None:
        with wave.open(archivo_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(wav_audio_data.getvalue())

def get_transcription(audio_path: str): 
    # Convertir audio a 16kHz para Whisper
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)  # Ajustar longitud del audio
    
    # Transcripción del audio en el idioma detectado
    with st.spinner(content_languages_dict[st.session_state.language_selected]['Transcribing and sending...'], show_time=True):
        result = st.session_state.model.transcribe(audio_path, language=st.session_state.language_selected)
    
    return result['text']

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file


cols = st.columns([1, 1, 5, 1, 1])
app = cols[2]

with cols[1]:
    st.markdown(f"""
                <img style="width:140px; height:110px; margin-top: 5px; margin-bottom: 0px; margin-left: 30px;" src="data:image/png;base64,{get_image_base64(f'{image_path}/odiseia-logo-white.png')}" alt="Imagen Semitransparente">
            """
            ,unsafe_allow_html=True)
    
with cols[3]:
    st.markdown(f"""
                <img style="width:170px; height:86px; margin-top: 20px; margin-bottom: 0px; margin-left: 20px;" src="data:image/png;base64,{get_image_base64(f'{image_path}/mioti_data&ai_logo_white_text_final.png')}" alt="Imagen Semitransparente">
            """
            ,unsafe_allow_html=True)
    
with app:

    header = st.container()

    # with st.popover('💡 info'):
    #     st.write("Aquí podemos añadir información.")

    body = st.container()

    with header:
        header_html = f"""
            <div class="header">
                <h1 style='text-align: center;padding:0px; color:#f9f8f7;'>TramitEasy</h1>
            </div>
            """
        st.markdown(header_html,unsafe_allow_html=True)


    with body:

        if 'language_selected' not in st.session_state:
            language_selector = st.selectbox(label="", options=list(languages_dict.keys()), index=None, placeholder="🌐 Select a Language")

            # st.image(image_path+'/mioti_data&ai_logo.png',use_container_width=True)


            st.markdown(f"""
                <img style="width:600px; height:270px; 
                    margin-top: 90px; margin-left: 150px; opacity: 0.1;" 
                    src="data:image/png;base64,{get_image_base64(f'{image_path}/mapa-mundi-black.png')}" 
                    alt="Imagen Semitransparente">
            """, unsafe_allow_html=True)

            if language_selector:
                st.session_state.language_selected = languages_dict[language_selector]
                st.rerun()
        
        else:
            if 'model' not in st.session_state:
                with st.spinner(content_languages_dict[st.session_state.language_selected]['Loading language model...'], show_time=True):
                    st.session_state.model = whisper.load_model("small")
                    if st.session_state.language_selected == "zh":
                        st.session_state.agent = RagAgent("hackaton", "zh-CN", "log_h.txt")
                    else:
                        st.session_state.agent = RagAgent("hackaton", st.session_state.language_selected, "log_h.txt")
                    st.session_state.workflow = st.session_state.agent.setup_workflow()
                    st.session_state.include_system_prompt = True


            selected_language_name = {v: k for k, v in languages_dict.items()}[st.session_state.language_selected]
            st.write("")
            st.write(content_languages_dict[st.session_state.language_selected]["🌐 Language selected: "], selected_language_name)

            chat_history_block = st.container()
            chat_input_block = st.container()

            def get_chat():
                with chat_history_block:
                    for msg in st.session_state.messages:
                        st.chat_message(msg["role"]).write(msg["content"])
                
                    audio_file = text_to_speech(msg["content"], lang=st.session_state.language_selected)  # Generar el audio
                    with st.expander(content_languages_dict[st.session_state.language_selected]["🔊 Answer"]):  # Expander con botón
                        st.audio(audio_file, format="audio/mp3")  # Reproducir audio
                        os.remove(audio_file)

            if "messages" not in st.session_state:
                # st.session_state["messages"] = [{"role": "assistant", "avatar": image_path+'/escudo-de-espana.jpg', "content": "How can I help you?"}]
                st.session_state["messages"] = [{"role": "assistant", "content": content_languages_dict[st.session_state.language_selected]["How can I help you?"]}]
                get_chat()

            with chat_input_block:

                input_mode = st.segmented_control(
                    "", content_languages_dict[st.session_state.language_selected]["Response Modes"], selection_mode="single", default=content_languages_dict[st.session_state.language_selected]["Response Modes"][0]
                )

                if 'input_mode' not in st.session_state:
                    st.session_state.input_mode = input_mode
                if st.session_state.input_mode != input_mode:
                    st.session_state.input_mode = input_mode
                    get_chat()

                if input_mode == content_languages_dict[st.session_state.language_selected]["Response Modes"][0]: #Text
                    if prompt := st.chat_input(placeholder=content_languages_dict[st.session_state.language_selected]["Your message"]):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        if st.session_state.include_system_prompt:
                            messages = [SystemMessage(content = st.session_state.agent.system_prompt), HumanMessage(content = prompt)]
                            st.session_state.include_system_prompt = False
                        else:
                            messages = [HumanMessage(content=prompt)]    

                        #result: historial completo incluyendo system prompt (Los input del usuario incluyen el contexto de RAG)
                        #llm_output: ultima respuesta
                        response = st.session_state.workflow.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})
                        result = response["messages"][-1].content
                        if st.session_state.agent.language != "es":
                            result = st.session_state.agent.output_translator.translate(result)
                                                
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        get_chat()

                elif input_mode == content_languages_dict[st.session_state.language_selected]["Response Modes"][1]: #Audio
                    get_audio()
                    if Path(archivo_wav).exists():
                        prompt = get_transcription(archivo_wav)
                        os.remove(archivo_wav)
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        if st.session_state.include_system_prompt:
                            messages = [SystemMessage(content = st.session_state.agent.system_prompt), HumanMessage(content = prompt)]
                            st.session_state.include_system_prompt = False
                        else:
                            messages = [HumanMessage(content=prompt)]    

                        #result: historial completo incluyendo system prompt (Los input del usuario incluyen el contexto de RAG)
                        #llm_output: ultima respuesta
                        response = st.session_state.workflow.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})
                        result = response["messages"][-1].content
                        if st.session_state.agent.language != "es":
                            result = st.session_state.agent.output_translator.translate(result)
                                                
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        get_chat()

    footer = """
    <div class="sidebar-footer" style="margin-top: 20px; margin-bottom: -75px; color: white;">
        Made with <span class="heart">&#10084;</span> by <a href="https://dataconsulting.mioti.es/es/">MIOTI</a>
    </div>
    """
    st.markdown(footer,unsafe_allow_html=True)