import streamlit as st
import whisper
import base64
import io
import soundfile as sf
import numpy as np
import librosa

st.title("Grabación y Transcripción Directa de Audio")


uploaded_file = st.audio_input("Graba o sube tu audio")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # Previsualiza el audio

    audio_bytes = uploaded_file.read()

    # Crea un buffer a partir de los bytes
    audio_buffer = io.BytesIO(audio_bytes)

    # # Usa soundfile para leer el audio y obtener el array y su tasa de muestreo
    # audio_array, sample_rate = sf.read(audio_buffer)

    # # Re-muestrea a 16 kHz si es necesario (Whisper requiere 16 kHz)
    # if sample_rate != 16000:
    #     audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    # # Si el audio es estéreo, conviértelo a mono (promediando los canales)
    # if len(audio_array.shape) > 1:
    #     audio_array = np.mean(audio_array, axis=1)

    # # Asegúrate de que el array sea de tipo float32
    
    # audio_array = audio_array.astype(np.float32)

    # # Carga el modelo de Whisper (por ejemplo, "turbo")
    # model = whisper.load_model("base")
    
    # Transcribe directamente pasando el array
    result = model.transcribe(audio_buffer)
    ##########################################################
    
    
        # load audio and pad/trim it to fit 30 seconds

    audio = whisper.pad_or_trim(audio_array)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    print(f"Detected language: {probs}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)        
    

    st.subheader("Transcripción:")
    st.write(result.text)
    #st.write(result["text"])
    