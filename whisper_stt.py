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
    # Load the Whisper model (you can change 'base' to another model depending on performance)
    # audio_data, sr = librosa.load(file_path, sr=None)
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
    transcription = whisper_pipe(file_path)
    print(transcription)
  
    return transcription['text']


import tempfile
def transcribe_audio(uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    # Pass the file path to the Whisper pipeline
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
    transcription = whisper_pipe(file_path)
    return transcription['text']








