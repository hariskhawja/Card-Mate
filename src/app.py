from fastapi import FastAPI
from inference import InferencePipeline
import cv2
import threading
import time

import os
from dotenv import load_dotenv
from pathlib import Path
import speech_recognition as sr


# ---------- Configuration ----------
app = FastAPI()

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

latest_result = {} # Store latest results so we can serve them over HTTP

pipeline = None
pipeline_thread = None


# ---------- Helpers ----------
def my_sink(result, video_frame):
    global latest_result
    latest_result = result
    if result.get("output_image"):
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    # print(result)

def speech_loop():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)  # optional, helps with background noise
    while True:
        with mic as source:
            try:
                audio = r.listen(source, timeout=5)  # waits max 5s for speech
                text = r.recognize_google(audio)

                if text.lower() == 'new card':
                    print(latest_result)
                    # You could trigger some other logic here

                if text.lower() == 'new hand':
                    res = get_latest_result()
                    print(res)

            except sr.WaitTimeoutError:
                # No speech detected within timeout
                pass
            except sr.UnknownValueError:
                # Speech was unintelligible
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        time.sleep(1)  # Delay between listening cycles


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
    all_data.sort(key=lambda x: x[0][0])  # Sort x by x values

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