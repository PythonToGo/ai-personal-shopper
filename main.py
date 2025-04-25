# Entry point for AI Personal Shopper pipeline
from models.detector import ObjectDetector
from models.classifier import ImageClassifier
import os
from PIL import Image


def main():
    detector = ObjectDetector(model_path="yolov8n.pt")
    classifier = ImageClassifier(model_name="google/vit-base-patch16-224")
    
    image_path = "data/raw/sample.jpg"
    results = detector.predict(image_path, conf=0.5)
    detections = detector. extract_boxes(results)
    
    print("Detections: ", detections)
    
    os.makedirs("data/processed/crops", exist_ok=True)
    image = Image.open(image_path)
    
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = f"data/processed/crops/crop_{idx}.jpg"
        crop.save(crop_path)
        
        label = classifier.predict(crop_path)
        print(f"Crop {idx}: Predicted label = {label}")

if __name__ == "__main__":
    main()