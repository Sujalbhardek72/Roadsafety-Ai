from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

app = FastAPI(title="RoadSafety AI API")

# Allow Lovable/React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/yolo26_best.pt"

model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "RoadSafety AI",
        "model": "YOLO Pothole Detection"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name

    try:
        results = model.predict(
            source=temp_path,
            conf=0.25,
            imgsz=320,
            device="cpu",
            verbose=False
        )

        result = results[0]

        pothole_count = len(result.boxes)

        confidences = [
            float(conf)
            for conf in result.boxes.conf
        ]

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        if pothole_count >= 5:
            severity = "High"
        elif pothole_count >= 2:
            severity = "Medium"
        elif pothole_count == 1:
            severity = "Low"
        else:
            severity = "None"

        return {
            "success": True,
            "pothole_count": pothole_count,
            "confidence_score": round(average_confidence, 3),
            "severity": severity,
            "detections": [
                {
                    "confidence": round(float(conf), 3)
                }
                for conf in confidences
            ]
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)