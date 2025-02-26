import os
from pathlib import Path
import streamlit as st
import wave
import whisper

archivo_wav = "grabacion.wav"
if Path(archivo_wav).exists():
    os.remove(archivo_wav)
samplerate = 44100  # Frecuencia de muestreo (Hz)

with st.spinner('Cargando modelo de reconocimiento de voz...', show_time=True):
    if 'model' not in st.session_state:
        st.session_state.model = whisper.load_model("small")

def get_audio():
    wav_audio_data = st.audio_input("Record a voice message")
    if wav_audio_data is not None:
        with wave.open(archivo_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(wav_audio_data.getvalue())

def detectar_idioma_y_transcribir(audio_path: str): 
    # Convertir audio a 16kHz para Whisper
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)  # Ajustar longitud del audio
    #audio = whisper.resample_audio(audio, 16000)  # Asegurarse de que la frecuencia de muestreo sea 16kHz
    mel = whisper.log_mel_spectrogram(audio).to(st.session_state.model.device)
    
    st.write('Detectando idioma...')
    _, probs = st.session_state.model.detect_language(mel)
    
    idioma_detectado = max(probs, key=probs.get)
    probabilidad = probs[idioma_detectado]
    
    # Transcripción del audio en el idioma detectado
    st.write('Transcribiendo...')
    result = st.session_state.model.transcribe(audio_path, language=idioma_detectado)
    
    return idioma_detectado, probabilidad, result['text']

get_audio()
if Path(archivo_wav).exists():
    idioma, prob, transcripcion = detectar_idioma_y_transcribir(archivo_wav)
    st.write(f"Idioma detectado: {idioma} (Confianza: {prob:.2%})")
    st.write(f"Transcripción: {transcripcion}")
    os.remove(archivo_wav)