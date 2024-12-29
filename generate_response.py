import pyttsx3

def generate_voice_response(text: str, lang="en"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1)  # (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()
