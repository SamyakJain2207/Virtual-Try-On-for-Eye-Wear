```markdown
# 🕶️ Virtual Try-On for Eyewear: AI-Powered Face Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-blueviolet.svg)
![React](https://img.shields.io/badge/React-UI-61dafb.svg)

An end-to-end, full-stack application that fuses Computer Vision, MediaPipe, and Deep Learning to simulate real-world eyewear fittings. Built to be a scalable and highly responsive intelligent system, it accurately maps facial geometry to recommend and dynamically overlay frames in real-time.

## 🌟 Key Features
* **Custom CNN Architecture**: A fine-tuned EfficientNetB3 model trained on a custom-annotated slice of the UTKFace dataset for high-accuracy face shape classification (Round, Oval, Square, Heart, Oblong).
* **Real-Time Facial Landmark Detection**: Utilizes Google's MediaPipe Face Mesh to extract 468 dense 3D facial landmarks for precise frame anchoring.
* **Algorithmic Fallback System**: Implements a deterministic geometric mathematical fallback to ensure zero-downtime inference if the deep learning model encounters edge-case anomalies.
* **High-Performance API**: Asynchronous backend built with FastAPI and Uvicorn to handle rapid frame-by-frame inference requests.
* **Interactive Frontend**: A responsive, React-based user interface styled with TailwindCSS for seamless camera integration and product selection.

## 🏗️ System Architecture
1. **Frontend (`/frontend`)**: Captures webcam video feed, processes UI state with React, and sends standardized image payloads.
2. **Backend (`/backend`)**: FastAPI server that handles CORS, decodes base64 image streams, and routes data through the TensorFlow inference engine.
3. **ML Pipeline (`/model_training`)**: A fully reproducible, Jupyter-based training pipeline featuring a custom data annotation tool, auto-labeling scripts, and transfer learning workflows.

## 🚀 Quick Start (Local Development)

### 1. Setup the Backend
Navigate to the backend directory, create an isolated environment, and start the server:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
uvicorn backend:app --reload

```

*The server will run at `http://localhost:8000*`

### 2. Setup the Frontend

Serve the `index.html` file using any local server. If using Python:

```bash
cd frontend
python -m http.server 8080

```

Navigate to `http://localhost:8080` in your web browser. Grant camera permissions when prompted.

### 3. Training Pipeline (Optional)

To retrain the model, navigate to `/model_training`, place the UTKFace dataset in `data/input/utkface-manual/UTKFace`, and execute the `train.py` script.

## 🧠 Future Scope & Optimization

* Integration of Generative AI to map lighting and shadows onto the 2D frame overlays for hyper-realistic rendering.
* Migrating the TensorFlow inference engine to ONNX or TensorRT for reduced latency.
* Expanding the training pipeline to utilize distributed training across larger datasets.

```
```
