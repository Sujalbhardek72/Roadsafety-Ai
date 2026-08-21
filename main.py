from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import time

app = FastAPI(title="RoadSafety AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/yolo26_best.pt"

# Load model only once when server starts
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "RoadSafety AI",
        "model": "YOLO Pothole Detection"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    total_start = time.perf_counter()

    # -----------------------------
    # 1. Read uploaded image
    # -----------------------------
    read_start = time.perf_counter()

    contents = await file.read()

    read_time = time.perf_counter() - read_start

    print(f"[TIMING] File read: {read_time:.2f}s")
    print(f"[INFO] File size: {len(contents) / 1024:.2f} KB")

    # -----------------------------
    # 2. Convert image to PIL
    # -----------------------------
    image_start = time.perf_counter()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {
            "success": False,
            "error": "Invalid image file"
        }

    image_time = time.perf_counter() - image_start

    print(f"[TIMING] Image processing: {image_time:.2f}s")
    print(f"[INFO] Image size: {image.size}")

    # -----------------------------
    # 3. YOLO inference
    # -----------------------------
    inference_start = time.perf_counter()

    try:
        results = model.predict(
            source=image,
            conf=0.25,
            imgsz=320,
            device="cpu",
            verbose=False,
            max_det=20
        )

        result = results[0]

    except Exception as e:
        print(f"[ERROR] YOLO inference failed: {str(e)}")

        return {
            "success": False,
            "error": f"YOLO inference failed: {str(e)}"
        }

    inference_time = time.perf_counter() - inference_start

    print(f"[TIMING] YOLO inference: {inference_time:.2f}s")

    # -----------------------------
    # 4. Process detections
    # -----------------------------
    pothole_count = len(result.boxes)

    confidences = [
        float(conf)
        for conf in result.boxes.conf
    ]

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    # -----------------------------
    # 5. Calculate severity
    # -----------------------------
    if pothole_count >= 5:
        severity = "High"
    elif pothole_count >= 2:
        severity = "Medium"
    elif pothole_count == 1:
        severity = "Low"
    else:
        severity = "None"

    total_time = time.perf_counter() - total_start

    print(f"[RESULT] Potholes: {pothole_count}")
    print(f"[RESULT] Average confidence: {average_confidence:.3f}")
    print(f"[RESULT] Severity: {severity}")
    print(f"[TIMING] TOTAL: {total_time:.2f}s")

    # -----------------------------
    # 6. Return result
    # -----------------------------
    return {
        "success": True,
        "pothole_count": pothole_count,
        "confidence_score": round(average_confidence, 3),
        "severity": severity,
        "processing_time": round(total_time, 2),
        "detections": [
            {
                "confidence": round(float(conf), 3)
            }
            for conf in confidences
        ]
    }