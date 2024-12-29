import pyttsx3

def generate_voice_response(text: str, lang="en"):
    """
    Converts text to speech and plays it.
    You can use pyttsx3 for offline TTS. For more advanced voices, consider Coqui TTS.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()
