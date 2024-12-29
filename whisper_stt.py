import whisper
import os
import librosa
import torch
from transformers import pipeline

def transcribe_audio_raw(file_path: str) -> str:
    # file_path = "C:/Users/Lenovo/ML Notebooks/ERP Assistant/example.wav"
    # if not os.path.exists(file_path):
    #     print(f"File not found: {file_path}")
    # else:
    #     print("File found!")
    # audio_data, sr = librosa.load(file_path, sr=None)
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
    transcription = whisper_pipe(file_path)
    print(transcription)
  
    return transcription['text']


import tempfile
def transcribe_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
    transcription = whisper_pipe(file_path)
    return transcription['text']








