# FastAPI
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Roboflow Deps
from inference import InferencePipeline
import cv2
import time
import numpy as np
import json

# Threading/Async/env
import os
from dotenv import load_dotenv
import asyncio
import base64
from pathlib import Path
import threading

# TTS/Voice Commands
import speech_recognition as sr
import pyttsx3


# ---------- Configuration ----------
app = FastAPI()

# Load environment variables
load_dotenv()

# Mount static files directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "")), name="static")
dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

latest_result = {} # Store latest results so we can serve them over HTTP

# Global variables for models and processing
pipeline = None
current_frame = None
processing_active = False
pipeline_thread = None

player_cards = [] # PLAYER'S ACTIVE CARDS

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
    '10': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
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


# ---------- Helpers ----------
def process_frame_with_workflow(frame):
    """Process a single frame using the workflow approach (like app_original.py)"""
    global latest_result, ROBOFLOW_API_KEY
    
    try:
        # Save frame temporarily for processing
        temp_path = os.path.join(os.path.dirname(__file__), "temp_mobile_frame.jpg")
        cv2.imwrite(temp_path, frame)
                
        # Create a result holder
        frame_result = {}
        
        def capture_result(result, video_frame):
            nonlocal frame_result
            frame_result.update(result)
            # Also update global latest_result
            global latest_result
            latest_result.update(result)
        
        try:
            # Create pipeline for single frame processing
            pipeline = InferencePipeline.init_with_workflow(
                api_key=ROBOFLOW_API_KEY,
                workspace_name="storm-nmwcn",
                workflow_id="detect-count-and-visualize",
                video_reference=temp_path,
                max_fps=1,
                on_prediction=capture_result
            )
            
            # Start processing
            pipeline.start()
            
            # Wait for result with timeout - REDUCED FREQUENCY
            timeout_counter = 0
            while not frame_result and timeout_counter < 30:  # 3 second timeout
                time.sleep(0.2)  # Increased sleep time to reduce frequency
                timeout_counter += 1
            
            # Try to stop pipeline gracefully
            try:
                if hasattr(pipeline, 'stop'):
                    pipeline.stop()
                elif hasattr(pipeline, 'terminate'):
                    pipeline.terminate()
                elif hasattr(pipeline, 'close'):
                    pipeline.close()
            except Exception as stop_error:
                print(f"Warning: Could not stop pipeline gracefully: {stop_error}")
            
        except Exception as pipeline_error:
            print(f"Pipeline error: {pipeline_error}")
            frame_result = {}
        
        # Parse result
        if frame_result and 'count_objects' in frame_result:
            cards = parse_workflow_data(frame_result)
            result = {
                'count_objects': frame_result['count_objects'],
                'predictions': frame_result.get('predictions'),
                'cards': cards,
                'timestamp': time.time()
            }
        else:
            result = {
                'count_objects': 0,
                'predictions': None,
                'cards': [],
                'timestamp': time.time()
            }
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
        
    except Exception as e:
        print(f"Error processing frame with workflow: {e}")
        # Clean up temp file on error
        temp_path = os.path.join(os.path.dirname(__file__), "temp_mobile_frame.jpg")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            'count_objects': 0,
            'predictions': None,
            'cards': [],
            'timestamp': time.time(),
            'error': str(e)
        }

# Parse workflow data (adapted from app_original.py parse_data function)
def parse_workflow_data(results):
    """Parse workflow results similar to app_original.py"""
    try:
        length = results.get('count_objects', 0)
        predictions = results.get('predictions')
        
        if not predictions or length == 0:
            return []
        
        # Extract prediction data
        if hasattr(predictions, 'xyxy') and hasattr(predictions, 'data'):
            # This is the format from app_original.py
            coords = predictions.xyxy
            data = predictions.data
            class_names = data['class_name']
            
            all_data = [[a, b] for a, b in zip(coords, class_names)]
            all_data.sort(key=lambda x: x[0][0])  # Sort by x values
            
            cards = []
            for i in range(min(length, len(all_data))):
                loc, class_name = all_data[i]
                card = parse_card(class_name)
                if card not in cards:
                    cards.append(card)
                                    
            return cards
        else:
            # Fallback if format is different
            return []
            
    except Exception as e:
        print(f"Error parsing workflow data: {e}")
        return []
    
# Index 0 = Num, Index 1 = Suit
def parse_card(card):
    if len(card) == 3:
        num, suit = 10, card_suit_map[card[-1]]
    else:
        num, suit = card_number_map[card[0]], card_suit_map[card[1]]
    return [num, suit]
    
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

# Parse 'latest_results' for relevant data 
# def parse_data(results):
#     if 'cards' in results:
#         return results['cards']
    
#     if 'count_objects' not in results:
#         return []
        
#     length = results['count_objects']
#     if length == 0:
#         return []
        
#     # Handle direct model results
#     if hasattr(results.get('predictions'), 'predictions'):
#         cards = []
#         for prediction in results['predictions'].predictions:
#             class_name = prediction.class_name
#             card = parse_card(class_name)
#             if card not in cards:
#                 cards.append(card)
#         return cards
    
#     return []

def has_pair():
    global player_cards
    count = {}
    for card in player_cards:
        count[card[0]] = count.get(card[0], 0) + 1
        if count[card[0]] >= 2:
            return card[0]
    return False


# ---------- TTS/Voice Commands ----------
def get_player_cards_tts():
    global player_cards
    text = ""
    if not player_cards:
        text_to_speech("No cards found.")
        return
    else:
        for card in player_cards:
            try:

                num_word = number_word_map[card[0]]
                text += f"{num_word} of {card[1]}, "
            except Exception as e:
                print(f"Error, {e}")
                
        text_to_speech(text)
        return player_cards

def get_card_tts(card):
    global player_cards

    try:
        num_word = number_word_map[card[0]]
        text = f"{num_word} of {card[1]}, "
    except Exception as e:
        print(f"Error, {e}")
    
    if card in player_cards:
        text_to_speech(f"you already have {text}, in your hand")
    
    else:
        text_to_speech(f"you got {text}")
        player_cards.append(card)

def discard_card_tts(card):
    global player_cards
    num_word = number_word_map[card[0]]
    text = f"{num_word} of {card[1]}, "
    if card in player_cards:
        ind = player_cards.index(card)
        player_cards.remove(card)
        text_to_speech(f"Discarded {text} at position {ind+1} from your hand")
    else:
        text_to_speech(f"You don't have {text} in your hand to discard")

def discard_pair_tts(cards, indices):
    global player_cards
    if len(cards) != 2:
        return
    text_to_speech(f"You have a pair!")
    text_to_speech(f"Discard {number_word_map[cards[0][0]]} of {cards[0][1]} and {number_word_map[cards[1][0]]} of {cards[1][1]} at positions {indices[0]+1} and {indices[1]+1} from your hand")
    player_cards.remove(cards[0])
    player_cards.remove(cards[1])

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('volume', 1)
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()

def speech_loop():
    global player_cards
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)  # helps with background noise
    while True:
        while has_pair():
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
                audio = r.listen(source)
                text = r.recognize_google(audio)

                print(f"USER SAID: {text}")

                # Voice Commands
                if 'got a new card' in text.lower():
                    try:
                        res = get_latest_result()
                        print("RES", res)

                        get_card_tts(res[0])
                        print(player_cards)

                    except Exception as e:
                        print(f"Error: {e}")
                        print("empty res")


                elif 'got a new hand' in text.lower():
                    res = get_latest_result()
                    player_cards = res
                    get_player_cards_tts()

                elif 'get my hand' in text.lower():
                    print(player_cards)
                    get_player_cards_tts()

                elif 'remove card' in text.lower():
                    try:
                        res = get_latest_result()

                        discard_card_tts(res[0])  
                    except Exception as e:
                        text_to_speech("Could not detect card.")
                        print(f"Error: {e}")   
                
                elif 'this is a test' in text.lower():
                    text_to_speech("This is a test of the text to speech system.")

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def get_admin_page():
    """Serve FastAPI admin page at root path"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "admin.html"), "r", encoding='utf-8') as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading admin.html: {str(e)}")

# Serve static HTML files directly
@app.get("/admin.html", response_class=HTMLResponse)
async def get_admin_html_direct():
    try:
        with open(os.path.join(os.path.dirname(__file__), "admin.html"), "r", encoding='utf-8') as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading admin.html: {str(e)}")

@app.get("/camera.html", response_class=HTMLResponse)
async def get_camera_html_direct():
    try:
        with open(os.path.join(os.path.dirname(__file__), "camera.html"), "r", encoding='utf-8') as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading camera.html: {str(e)}")

# Serve the test page for continuous detection
@app.get("/test_continuous.html", response_class=HTMLResponse)
async def get_test_continuous_html():
    try:
        with open(os.path.join(os.path.dirname(__file__), "test_continuous.html"), "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading test_continuous.html: {str(e)}")

# FastAPI entrypoint upon startup
@app.on_event("startup")
def startup_event():
    print("Card-Mate Detection System started")
    print("Server ready to process mobile camera feeds")
    
    if ROBOFLOW_API_KEY:
        print("API key found, model will be initialized on first frame")
    else:
        print("Warning: ROBOFLOW_API_KEY not found in environment")

    # --- Start the speech recognition loop ---
    speech_thread = threading.Thread(target=speech_loop, daemon=True)
    speech_thread.start()

# FastAPI exitpoint upon shutdown
@app.on_event("shutdown")
def shutdown_event():
    global direct_model
    
    print("Shutting down Card-Mate Detection System")
    
    # Clean up references
    direct_model = None

# Simple test endpoint to verify server connectivity
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is running"}

# Health check endpoint
@app.get("/health")
def health_check():
    global direct_model, latest_result
    return {
        "status": "healthy",
        "mobile_processing": True,
        "model_ready": direct_model is not None,
        "latest_results_available": bool(latest_result),
        "timestamp": time.time()
    }

# Get current processing status
@app.get("/status")
def get_status():
    global direct_model, latest_result
    return {
        "message": "Card-Mate Mobile Detection System",
        "mobile_processing": True,
        "model_status": "Ready" if direct_model is not None else "Initializing",
        "latest_results_count": latest_result.get('count_objects', 0) if latest_result else 0,
        "timestamp": time.time()
    }

# Test endpoint for workflow functionality
@app.get("/test-workflow")
def test_workflow():
    """Test the workflow processing with a simple image"""
    try:
        # Create a simple test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame = cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Process it
        result = process_frame_with_workflow(test_frame)
        
        return {
            "status": "success",
            "message": "Workflow test completed",
            "result": result,
            "api_key_present": bool(ROBOFLOW_API_KEY)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Workflow test failed: {str(e)}",
            "api_key_present": bool(ROBOFLOW_API_KEY)
        }
      
@app.get("/backend-results")
def get_latest_result():
    global latest_result
    print("LATEST RESULT", latest_result)
    if 'count_objects' not in latest_result:
        return {"error": "No results available yet."}

    prev = []

    threshold = 0 # We want the results to be the same for at least 3 ticks before moving on
    timeout = 30 # If there's nothing that's the same within 30 ticks, then move on
    i = 0
    t = 0

    cur = parse_data(latest_result)
    return cur
    # while t < timeout:
    #     # print("i: ", i)
    #     # print(f"Latest Result: {latest_result}")
    #     # cur = parse_data(latest_result)
    #     if i >= threshold:
    #         return cur
    #     # print("Current Card: ", cur)
    #     if cur != prev:
    #         i = 0
    #         prev = cur
    #         continue
    #     prev = cur
    #     i += 1
    #     t += 1
    #     time.sleep(0.1)
    # return {"error": "Timeout."}

# Main endpoint to grab model results
@app.get("/results")
def get_frontend_latest_results():
    global latest_result
    if 'count_objects' not in latest_result:
        return {"error": "No results available yet."}
      
    return {
        "count": latest_result['count_objects'],
        "cards": latest_result.get('cards', []),
        "timestamp": latest_result.get('timestamp', time.time())
    }

# Endpoint for mobile devices to upload camera frames
@app.post("/upload-frame")
async def upload_frame(file: UploadFile = File(...)):
    """Process uploaded frame from mobile camera"""
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image format"}
        
        # Process frame
        result = process_frame_with_workflow(frame)
        
        return {
            "count": result['count_objects'],
            "cards": result.get('cards', []),
            "timestamp": result.get('timestamp', time.time()),
            "processing_time": time.time() - result.get('timestamp', time.time())
        }
        
    except Exception as e:
        return {"error": f"Frame processing error: {str(e)}"}

# WebSocket endpoint for real-time camera streaming
@app.websocket("/ws/camera")
async def websocket_camera_stream(websocket: WebSocket):
    await websocket.accept()
    print("Mobile camera WebSocket connection established")
    
    try:
        while True:
            # Receive frame data from mobile
            data = await websocket.receive_text()
            
            try:
                # Parse the received data
                frame_data = json.loads(data)
                
                if frame_data.get('type') == 'frame' and 'data' in frame_data:
                    # Decode base64 image
                    image_data = base64.b64decode(frame_data['data'])
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Process the frame
                        result = process_frame_with_workflow(frame)
                        
                        # Send back results
                        response = {
                            "count": result['count_objects'],
                            "cards": result.get('cards', []),
                            "timestamp": result.get('timestamp', time.time()),
                            "frame_timestamp": frame_data.get('timestamp', time.time())
                        }
                        
                        await websocket.send_json(response)
                    else:
                        await websocket.send_json({"error": "Invalid frame data"})
                else:
                    await websocket.send_json({"error": "Invalid frame format - expected type:'frame' and data"})
                        
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON data"})
            except Exception as process_error:
                await websocket.send_json({"error": f"Processing error: {str(process_error)}"})
                
    except Exception as e:
        print(f"Mobile camera WebSocket error: {e}")
    finally:
        print("Mobile camera WebSocket connection closed")


# WebSocket endpoint for real-time detection updates (legacy support)
@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established for real-time detection updates")
    
    try:
        last_sent_result = None
        while True:
            # Send latest results if they've changed
            if latest_result and latest_result != last_sent_result:
                try:
                    # Format the result for frontend
                    formatted_result = {
                        "count": latest_result.get('count_objects', 0),
                        "timestamp": latest_result.get('timestamp', time.time()),
                        "cards": latest_result.get('cards', []),
                        "data": latest_result
                    }
                    
                    await websocket.send_json(formatted_result)
                    last_sent_result = latest_result.copy() if isinstance(latest_result, dict) else latest_result
                    
                except Exception as send_error:
                    print(f"Error sending WebSocket data: {send_error}")
                    break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)  # 10 FPS update rate
            
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        print("WebSocket connection closed")


# ---------- Activate uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)