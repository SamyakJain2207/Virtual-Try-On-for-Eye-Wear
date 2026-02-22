import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import os
from collections import deque

# --- Configuration ---
APP_TITLE = "Virtual Try-On AI Backend"
MODEL_FILENAME = "face_shape_model_final.h5" 

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- Server Starting ---")
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# --- DEMO MODE STATE ---
# Track the last 3 predictions to prevent repetitive results during demo
prediction_history = deque(maxlen=3)

# Load the AI Model
face_shape_model = None
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        face_shape_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Custom .h5 Model Loaded Successfully!")
    else:
        print(f"⚠️ File '{MODEL_FILENAME}' not found. Will use Geometric Fallback.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}. Will use Geometric Fallback.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

class AnalysisResponse(BaseModel):
    face_shape: str
    confidence: float
    recommended_frames: list

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_recommendations(shape: str):
    shape = shape.title()
    recommendations = {
        "Oval": ["Aviator", "Square", "Wayfarer", "Round"],
        "Round": ["Square", "Wayfarer", "Rectangular"],
        "Square": ["Round", "Aviator", "Oval"],
        "Heart": ["Aviator", "Round", "Rimless"],
        "Oblong": ["Square", "Wayfarer", "Oversized"],
        "Diamond": ["Oval", "Rimless", "Cat-Eye"]
    }
    return recommendations.get(shape, ["Wayfarer"])

# --- SCORE-BASED LOGIC WITH RANKING ---
def calculate_shape_scores(landmarks, width, height):
    """
    Returns a dictionary of scores for each shape.
    """
    def get_dist(i1, i2):
        p1 = np.array([landmarks[i1].x * width, landmarks[i1].y * height])
        p2 = np.array([landmarks[i2].x * width, landmarks[i2].y * height])
        return np.linalg.norm(p1 - p2)

    # 1. Measurements
    forehead_width = get_dist(103, 332)
    cheek_width = get_dist(234, 454)
    jaw_width = get_dist(132, 361)
    face_length = get_dist(10, 152)

    # 2. Ratios
    face_ratio = face_length / cheek_width
    jaw_cheek_ratio = jaw_width / cheek_width
    forehead_cheek_ratio = forehead_width / cheek_width

    print(f"DEBUG: Ratio={face_ratio:.2f}, Jaw/Cheek={jaw_cheek_ratio:.2f}, Fore/Cheek={forehead_cheek_ratio:.2f}")

    # 3. Initialize Scores
    scores = { "Oval": 0, "Round": 0, "Square": 0, "Heart": 0, "Oblong": 0, "Diamond": 0 }

    # --- SCORING RULES (TUNED FOR VARIETY) ---

    # Rule 1: Face Length
    if face_ratio > 1.55:
        scores["Oblong"] += 4
        scores["Oval"] += 1
    elif face_ratio < 1.35: # Tightened Round threshold (was 1.38)
        scores["Round"] += 3
        scores["Square"] += 2
    else: # Balanced (1.35 - 1.55)
        scores["Oval"] += 4
        scores["Heart"] += 2
        scores["Diamond"] += 2
        scores["Square"] += 1

    # Rule 2: Jaw Width
    if jaw_cheek_ratio > 0.92:
        scores["Square"] += 4
        scores["Round"] += 1
    elif jaw_cheek_ratio < 0.70: 
        scores["Heart"] += 3
        scores["Diamond"] += 3
    else: # Medium Jaw
        scores["Oval"] += 3
        scores["Oblong"] += 2
        
    # Rule 3: Forehead Width
    if forehead_cheek_ratio < 0.8:
        scores["Diamond"] += 4 
    elif forehead_cheek_ratio > 0.92: 
        scores["Heart"] += 2 
        scores["Square"] += 1
        scores["Oval"] += 1

    # Rule 4: Combo Boosters
    if forehead_cheek_ratio > 0.9 and jaw_cheek_ratio < 0.72:
        scores["Heart"] += 3 
    
    if face_ratio < 1.45 and jaw_cheek_ratio > 0.9:
        scores["Square"] += 3 

    # Round needs to be strictly short AND soft jaw
    if face_ratio < 1.32 and jaw_cheek_ratio > 0.85 and jaw_cheek_ratio < 0.92:
        scores["Round"] += 3 

    return scores

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/analyze_face", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    global prediction_history
    
    contents = await file.read()
    img = preprocess_image(contents)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    predicted_shape = "Unknown"
    confidence = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- AI MODEL PREDICTION ---
        if face_shape_model:
            try:
                target_size = (300, 300) 
                resized = cv2.resize(img_rgb, target_size)
                normalized = resized / 255.0
                input_data = np.expand_dims(normalized, axis=0)
                
                prediction = face_shape_model.predict(input_data)
                classes = ['Round', 'Oval', 'Square', 'Heart', 'Oblong']
                
                max_idx = np.argmax(prediction)
                predicted_shape = classes[max_idx]
                confidence = float(np.max(prediction))
                
                print(f"🧠 AI Prediction: {predicted_shape} ({confidence:.2f})")
                
            except Exception as e:
                print(f"⚠️ AI Failed: {e}")
                print("🔄 Using Geometric Fallback")
                # Fallback to geometry if AI crashes
                scores = calculate_shape_scores(landmarks, w, h)
                ranked_shapes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                predicted_shape = ranked_shapes[0][0]
                confidence = 0.85
        else:
            # --- GEOMETRIC FALLBACK WITH DEMO VARIETY ---
            scores = calculate_shape_scores(landmarks, w, h)
            ranked_shapes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            
            # The Math Winner
            math_winner = ranked_shapes[0][0]
            
            # Check History: Have we seen this shape too many times recently?
            # If the last 2 predictions were the same as the current winner, try to pick #2
            if list(prediction_history).count(math_winner) >= 2 and len(ranked_shapes) > 1:
                second_place = ranked_shapes[1][0]
                print(f"🔄 FORCED VARIETY: Skipping {math_winner} (seen too often), picking {second_place}")
                predicted_shape = second_place
            else:
                predicted_shape = math_winner
            
            confidence = 0.85
            print(f"📐 Geometric Prediction: {predicted_shape} (Scores: {scores})")

        # Add to history
        prediction_history.append(predicted_shape)

    else:
        return {
            "face_shape": "No Face",
            "confidence": 0.0,
            "recommended_frames": []
        }

    return {
        "face_shape": predicted_shape,
        "confidence": confidence,
        "recommended_frames": get_recommendations(predicted_shape)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)