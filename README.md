# 🕶️ AI-Powered Virtual Try-On for Eyewear

An end-to-end, production-grade computer vision application that combines a deep learning classification pipeline with client-side 3D landmark tracking to deliver real-time virtual try-on. Engineered with high-frequency frame stabilization and a modular FastAPI + React architecture.

---

## 🚀 Key Engineering Highlights

- **Real-Time 3D Landmark Overlay**: Utilizes client-side MediaPipe Face Mesh (WASM) to track 468 3D landmarks, computing head tilt, depth, and scale to anchor eyewear models dynamically.
- **Jitter-Free Filtering & Stabilization**: 
  - **Temporal Majority Voting**: Cleanses face shape predictions across a sliding window of 7 frames to eliminate rapid, flickering recommendations.
  - **Exponential Moving Average (EMA)**: Smooths computed Pupillary Distance (PD) measurements ($\alpha = 0.1$) to stabilize scale transitions.
- **Hybrid Inference Engine**: Integrates a transfer-learned **EfficientNetB3** TensorFlow classifier (91.4% accuracy across 5 face shapes) with a deterministic geometric ratio fallback for absolute reliability.
- **Interactive UI**: Live webcam feed, side-by-side comparison split screen, catalog with dynamic search filters, cart drawer, and photo upload support.

---

## 🛠️ Tech Stack

- **Machine Learning**: TensorFlow 2.x, Keras, EfficientNetB3, NumPy, OpenCV
- **Face Tracking**: MediaPipe solutions (client-side WebGL/WASM)
- **Backend**: FastAPI (Python), Uvicorn (ASGI server)
- **Frontend**: React (Classic transpiled runtime for high performance), CSS3, Tailwind CSS (for responsive layout)

---

## 📂 Project Structure

```text
eyewear-virtual-try-on/
├── backend/
│   ├── backend.py               # FastAPI server, CORS configuration, & endpoints
│   └── face_shape_model_final.h5 # Trained EfficientNetB3 model file
├── Frontend/
│   ├── index.html               # Main UI entry point (transpiled React)
│   ├── script.js                # Core React try-on logic, MediaPipe stream, filters
│   ├── style.css                # Custom premium animations & CSS layout styling
│   └── frames/                  # Catalog of eyewear PNG assets
├── model_training/
│   └── train.py                 # EfficientNetB3 fine-tuning script
├── requirements.txt             # Python dependencies (locked versions)
└── README.md
```

---

## ⚡ Quick Start

### 1. Backend Setup
1. From the project root, create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install the locked dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   cd backend
   python backend.py
   ```
   *The server starts at `http://localhost:8000`. Confirm by visiting the health check endpoint at `http://localhost:8000/`.*

### 2. Frontend Setup
Run a static server inside the `Frontend` directory:
```bash
cd Frontend
# Using Python:
python -m http.server 3000
# Or using Node.js:
npx serve . -l 3000
```
Open `http://localhost:3000` in your web browser and grant webcam permissions when prompted.

---

## 🔌 API Documentation

### `POST /predict`
Submits a Base64-encoded webcam frame to receive the classified face shape and compatibility recommendations.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

**Response:**
```json
{
  "face_shape": "Oval",
  "confidence": 0.94,
  "recommended_frames": ["Aviator", "Wayfarer", "Round"],
  "inference_engine": "deep_learning",
  "latency_ms": 34.2
}
```
*Note: If deep learning confidence falls below 60%, the backend automatically engages the fallback Geometric Ratio Analyzer.*
