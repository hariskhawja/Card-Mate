from fastapi import FastAPI
from inference import InferencePipeline
import cv2
import threading
import time

import os
from dotenv import load_dotenv
from pathlib import Path
import speech_recognition as sr
import pyttsx3


# ---------- Configuration ----------
app = FastAPI()

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

latest_result = {} # Store latest results so we can serve them over HTTP

pipeline = None
pipeline_thread = None

player_cards = []

# ---------- Helpers ----------
def my_sink(result, video_frame):
    global latest_result
    latest_result = result
    if result.get("output_image"):
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    # print(result)

def speech_loop():
    global player_cards
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)  # optional, helps with background noise
    while True:
        while has_pair():
            print("Here")
            pair = has_pair()
            print(f"Found pair: {pair}")
            c = []
            indices = []
            for i in range(len(player_cards)):
                card = player_cards[i]
                if len(c) >= 2:
                    break
                if card[0] == pair:
                    c.append(card)
                    indices.append(i)
            discard_pair_tts(c, indices)
        with mic as source:
            try:
                audio = r.listen(source)  # waits max 5s for speech
                text = r.recognize_google(audio)

                print(text)

                if 'got a new card' in text.lower():
                    # print("new card")

                    try:
                        res = get_latest_result()

                        get_card_tts(res[0])

                    except Exception as e:
                        print(f"Error: {e}")                
                    print(player_cards)

                if 'got a new hand' in text.lower():
                    # print('new hand')

                    res = get_latest_result()
                    player_cards = res
                    get_player_cards_tts()

                if 'get my hand' in text.lower():
                    print(player_cards)
                    get_player_cards_tts()

                if 'remove card' in text.lower():
                    try:
                        res = get_latest_result()

                        discard_card_tts(res[0])  
                    except Exception as e:
                        text_to_speech("Could not detect card.")
                        print(f"Error: {e}")   

                
                if 'this is a test' in text.lower():
                    print('test')
                    text_to_speech("This is a test of the text to speech system.")

            except sr.WaitTimeoutError:
                # No speech detected within timeout
                pass
            except sr.UnknownValueError:
                # Speech was unintelligible
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        # time.sleep(0.2)  # Delay between listening cycles

        

def has_pair():
    count = {}
    for card in player_cards:
        count[card[0]] = count.get(card[0], 0) + 1
        if count[card[0]] >= 2:
            return card[0]
    return False

def get_player_cards_tts():
    text = ""
    if not player_cards:
        text_to_speech("No cards found.")
        return
    for card in player_cards:
        num_word = number_word_map[card[0]]
        text += f"{num_word} of {card[1]}, "
    text_to_speech(text)
    return player_cards

def get_card_tts(card):
    num_word = number_word_map[card[0]]
    text = f"{num_word} of {card[1]}, "
    
    if card in player_cards:
        text_to_speech(f"you already have {text}, in your hand")
    
    else:
        text_to_speech(f"you got {text}")
        player_cards.append(card)

def discard_card_tts(card):
    num_word = number_word_map[card[0]]
    text = f"{num_word} of {card[1]}, "
    if card in player_cards:
        ind = player_cards.index(card)
        player_cards.remove(card)
        text_to_speech(f"Discarded {text} at position {ind+1} from your hand")
    else:
        text_to_speech(f"You don't have {text} in your hand to discard")

def discard_pair_tts(cards, indices):
    if len(cards) != 2:
        return
    text_to_speech(f"You have a pair!")
    text_to_speech(f"Discard {number_word_map[cards[0][0]]} of {cards[0][1]} and {number_word_map[cards[1][0]]} of {cards[1][1]} at positions {indices[0]+1} and {indices[1]+1} from your hand")
    player_cards.remove(cards[0])
    player_cards.remove(cards[1])

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('volume', 1)  # Set to 100% volume (0.0 to 1.0)
    engine.setProperty('rate', 150)  # Set to 150 words per minute
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()

# ---------- FastAPI Endpoints ----------
# FastAPI entrypoint upon startup
@app.on_event("startup")
def startup_event():
    global pipeline, pipeline_thread
    pipeline = InferencePipeline.init_with_workflow(
        api_key=ROBOFLOW_API_KEY,
        workspace_name="storm-nmwcn",
        workflow_id="detect-count-and-visualize",
        video_reference=0,
        max_fps=30,
        on_prediction=my_sink
    )

    # Run pipeline in a separate thread so FastAPI can still respond
    pipeline_thread = threading.Thread(target=pipeline.start, daemon=True)
    pipeline_thread.start()

    # --- Start the speech recognition loop ---
    speech_thread = threading.Thread(target=speech_loop, daemon=True)
    speech_thread.start()

# FastAPI exitpoint upon shutdown
@app.on_event("shutdown")
def shutdown_event():
    if pipeline:
        pipeline.stop()

# Main endpoint to grab model results
@app.get("/results")
def get_latest_result():
    
    # print("latest_result: ", latest_result['count_objects'])
    if 'count_objects' not in latest_result:
        return {"error": "No results available yet."}

    prev = []

    threshold = 3 # We want the results to be the same for at least 3 ticks before moving on
    timeout = 30 # If there's nothing that's the same within 30 ticks, then move on
    i = 0
    t = 0
    while t < timeout:
        # print("i: ", i)
        # print(f"Latest Result: {latest_result}")
        if i >= threshold:
            return cur
        cur = parse_data(latest_result)
        # print("Current Card: ", cur)
        if cur != prev:
            i = 0
            prev = cur
            continue
        prev = cur
        i += 1
        t += 1
        time.sleep(0.1)
    return {"error": "Timeout."}


# Parse 'latest_results' for relevant data
def parse_data(results):
    length = results['count_objects']
    predictions = results['predictions']
    coords = predictions.xyxy
    data = predictions.data
    class_names = data['class_name']

    all_data = [[a, b] for a, b in zip(coords, class_names)]
    all_data.sort(key=lambda x: x[0][0], reverse=True)  # Sort x by x values

    cards = []
    
    for i in range(length):
        loc, class_name = all_data[i]
        card = parse_card(class_name)
        if card in cards:
            continue
        cards.append(card)

    return cards

# Mapping Numbers for each Card
card_number_map = { 
    'A': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,  # T for Ten
    'J': 11,  # Jack
    'Q': 12,  # Queen
    'K': 13,  # King
}

number_word_map = {
    1: "Ace",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Jack",
    12: "Queen",
    13: "King"
}


# Mapping Suits for each Card
card_suit_map = { 
    'C': 'Clubs',
    'D': 'Diamonds',
    'S': 'Spades',
    'H': 'Hearts',
}


# Index 0 = Num, Index 1 = Suit
def parse_card(card):
    if len(card) == 3:
        num, suit = 10, card_suit_map[card[-1]]
    else:
        num, suit = card_number_map[card[0]], card_suit_map[card[1]]
    return [num, suit]