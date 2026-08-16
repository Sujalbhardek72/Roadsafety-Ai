from ultralytics import YOLO

MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\Roadsafety-AI\models\yolo26_best.pt"

model = YOLO(MODEL_PATH)

results = model.predict(
    source="road_video.mp4",
    save=True,
    conf=0.25,
    stream=False
)

print("Video detection completed successfully!")