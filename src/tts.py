import pyttsx3
engine = pyttsx3.init()
engine.setProperty('volume', 1) # Set to 80% volume (0.0 to 1.0)
engine.setProperty('rate', 150) # Set to 150 words per minute
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

engine.say("That is a 1 of spades!")
engine.runAndWait()