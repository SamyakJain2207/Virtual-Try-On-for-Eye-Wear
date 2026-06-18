import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import os

# --- Configuration ---
APP_TITLE = "Virtual Try-On AI Backend"
MODEL_FILENAME = "face_shape_model_final.h5"

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- Server Starting ---")
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Load the AI Model
face_shape_model = None
try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        face_shape_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[SUCCESS] Custom .h5 Model Loaded Successfully!")
    else:
        print(f"[WARNING] File '{MODEL_FILENAME}' not found. Will use Geometric Fallback.")
except Exception as e:
    print(f"[WARNING] Error loading model: {e}. Will use Geometric Fallback.")

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

# FIX 4: Recommendation IDs now exactly match GLASSES_DB IDs in script.js
def get_recommendations(shape: str):
    shape = shape.title()
    recommendations = {
        "Oval":    ["Aviator", "Wayfarer", "Square Thick", "Round Thin"],
        "Round":   ["Square Thick", "Wayfarer", "Square Tortoise", "Black Gold"],
        "Square":  ["Round Gold", "Aviator", "Classic Round", "Sun Round"],
        "Heart":   ["Aviator", "Round Gold", "Round Thin", "Sun Aviator"],
        "Oblong":  ["Square Thick", "Wayfarer", "Black Gold", "Square Tortoise"],
        "Diamond": ["Classic Round", "Round Thin", "CatEye Tortoise", "CatEye White"]
    }
    return recommendations.get(shape, ["Wayfarer", "Aviator"])

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
    cheek_width    = get_dist(234, 454)
    jaw_width      = get_dist(132, 361)
    face_length    = get_dist(10, 152)

    # 2. Ratios
    face_ratio             = face_length / cheek_width
    jaw_cheek_ratio        = jaw_width   / cheek_width
    forehead_cheek_ratio   = forehead_width / cheek_width

    print(f"DEBUG: Ratio={face_ratio:.2f}, Jaw/Cheek={jaw_cheek_ratio:.2f}, Fore/Cheek={forehead_cheek_ratio:.2f}")

    # 3. Initialize Scores
    scores = { "Oval": 0, "Round": 0, "Square": 0, "Heart": 0, "Oblong": 0, "Diamond": 0 }

    # Rule 1: Face Length
    if face_ratio > 1.55:
        scores["Oblong"] += 4
        scores["Oval"]   += 1
    elif face_ratio < 1.35:
        scores["Round"]  += 3
        scores["Square"] += 2
    else:
        scores["Oval"]    += 4
        scores["Heart"]   += 2
        scores["Diamond"] += 2
        scores["Square"]  += 1

    # Rule 2: Jaw Width
    if jaw_cheek_ratio > 0.92:
        scores["Square"] += 4
        scores["Round"]  += 1
    elif jaw_cheek_ratio < 0.70:
        scores["Heart"]   += 3
        scores["Diamond"] += 3
    else:
        scores["Oval"]   += 3
        scores["Oblong"] += 2

    # Rule 3: Forehead Width
    if forehead_cheek_ratio < 0.8:
        scores["Diamond"] += 4
    elif forehead_cheek_ratio > 0.92:
        scores["Heart"]  += 2
        scores["Square"] += 1
        scores["Oval"]   += 1

    # Rule 4: Combo Boosters
    if forehead_cheek_ratio > 0.9 and jaw_cheek_ratio < 0.72:
        scores["Heart"] += 3

    if face_ratio < 1.45 and jaw_cheek_ratio > 0.9:
        scores["Square"] += 3

    if face_ratio < 1.32 and jaw_cheek_ratio > 0.85 and jaw_cheek_ratio < 0.92:
        scores["Round"] += 3

    return scores

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/analyze_face", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w, _ = img.shape
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results  = face_mesh.process(img_rgb)

    predicted_shape = "Unknown"
    confidence      = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- AI MODEL PREDICTION ---
        if face_shape_model:
            try:
                target_size = (300, 300)
                resized = cv2.resize(img_rgb, target_size)

                # FIX 2: Model has InputLayer → Functional (EfficientNetB3) with no
                # preprocessing layer baked in, so pass raw float32 [0, 255] values.
                # The EfficientNet preprocess_input inside the Functional block
                # expects this range and will scale internally.
                input_data = np.expand_dims(resized.astype(np.float32), axis=0)

                # FIX 3: verbose=0 suppresses per-call progress bar in terminal
                prediction = face_shape_model.predict(input_data, verbose=0)

                # FIX 1: Train labels were lowercase; title-case only for display
                classes = ['round', 'oval', 'square', 'heart', 'oblong']
                max_idx         = np.argmax(prediction)
                predicted_shape = classes[max_idx].title()
                confidence      = float(np.max(prediction))

                print(f"[AI] Prediction: {predicted_shape} ({confidence:.2f})")

            except Exception as e:
                print(f"[AI Failed] {e}")
                print("[FALLBACK] Using Geometric Fallback")
                scores          = calculate_shape_scores(landmarks, w, h)
                ranked_shapes   = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                predicted_shape = ranked_shapes[0][0]
                confidence      = 0.85

        else:
            # FIX 5: Geometric fallback — removed forced-variety demo logic,
            # just pick the top scorer cleanly.
            scores          = calculate_shape_scores(landmarks, w, h)
            ranked_shapes   = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            predicted_shape = ranked_shapes[0][0]
            confidence      = 0.85
            print(f"[GEOMETRIC] Prediction: {predicted_shape} (Scores: {scores})")

    else:
        return {
            "face_shape":          "No Face",
            "confidence":          0.0,
            "recommended_frames":  []
        }

    return {
        "face_shape":          predicted_shape,
        "confidence":          confidence,
        "recommended_frames":  get_recommendations(predicted_shape)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)