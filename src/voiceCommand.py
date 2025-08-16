import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    text = r.recognize_google(audio)
    print(f"YOU SAID {text}")

    if text.lower() == 'new card':
        print("NEW CARD")