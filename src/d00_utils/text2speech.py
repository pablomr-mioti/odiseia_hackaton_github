import streamlit as st
from gtts import gTTS
import os

# Función para convertir texto en voz
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file

# Inicializar el historial del chat en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Simulación del chatbot (sustituye por tu lógica real)
user_input = st.text_input("Escribe tu mensaje:")
if user_input:
    bot_response = f"Has dicho: {user_input}"  # Aquí pondrías la respuesta real del chatbot
    st.session_state.messages.append(("👤", user_input))
    st.session_state.messages.append(("🤖", bot_response))

# Mostrar el historial del chat
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.write(message)
        
        # Solo para respuestas del bot, añadir botón de voz
        if role == "🤖":
            audio_file = text_to_speech(message)  # Generar el audio
            with st.expander("🔊 Escuchar respuesta"):  # Expander con botón
                st.audio(audio_file, format="audio/mp3")  # Reproducir audio
