from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from inference import InferencePipeline
import cv2
import time
import os
import numpy as np
import json
from dotenv import load_dotenv
import asyncio
import base64

# Load environment variables
load_dotenv()

# ---------- Configuration ----------
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "")), name="static")

latest_result = {} # Store latest results so we can serve them over HTTP

# Global variables for models and processing
pipeline = None
current_frame = None
processing_active = False
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# ---------- Helpers ----------
def process_frame_with_workflow(frame):
    """Process a single frame using the workflow approach (like app_original.py)"""
    global latest_result, ROBOFLOW_API_KEY
    
    try:
        # Save frame temporarily for processing
        temp_path = os.path.join(os.path.dirname(__file__), "temp_mobile_frame.jpg")
        cv2.imwrite(temp_path, frame)
        
        # Use the workflow approach directly with the image
        from inference import InferencePipeline
        
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
            
            # Wait for result with timeout
            timeout_counter = 0
            while not frame_result and timeout_counter < 30:  # 3 second timeout
                time.sleep(0.1)
                timeout_counter += 1
            
            # Try to stop pipeline gracefully
            try:
                # Try different methods to stop the pipeline
                if hasattr(pipeline, 'stop'):
                    pipeline.stop()
                elif hasattr(pipeline, 'terminate'):
                    pipeline.terminate()
                elif hasattr(pipeline, 'close'):
                    pipeline.close()
                else:
                    # Pipeline will be garbage collected
                    del pipeline
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

# Parse 'latest_results' for relevant data 
def parse_data(results):
    if 'cards' in results:
        return results['cards']
    
    if 'count_objects' not in results:
        return []
        
    length = results['count_objects']
    if length == 0:
        return []
        
    # Handle direct model results
    if hasattr(results.get('predictions'), 'predictions'):
        cards = []
        for prediction in results['predictions'].predictions:
            class_name = prediction.class_name
            card = parse_card(class_name)
            if card not in cards:
                cards.append(card)
        return cards
    
    return []

# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def get_admin_page():
    """Serve FastAPI admin page at root path"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "admin.html"), "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading admin.html: {str(e)}")

# Serve static HTML files directly
@app.get("/admin.html", response_class=HTMLResponse)
async def get_admin_html_direct():
    try:
        with open(os.path.join(os.path.dirname(__file__), "admin.html"), "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading admin.html: {str(e)}")

@app.get("/camera.html", response_class=HTMLResponse)
async def get_camera_html_direct():
    try:
        with open(os.path.join(os.path.dirname(__file__), "camera.html"), "r") as file:
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
    
    # Initialize the direct model for mobile processing
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key:
        print("API key found, model will be initialized on first frame")
    else:
        print("Warning: ROBOFLOW_API_KEY not found in environment")

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

# Main endpoint to grab model results
@app.get("/results")
def get_latest_result():
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

# Index 0 = Num, Index 1 = Suit
def parse_card(card):
    if len(card) == 3:
        num, suit = 10, card_suit_map[card[-1]]
    else:
        num, suit = card_number_map[card[0]], card_suit_map[card[1]]
    return [num, suit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
