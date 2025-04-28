from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.device = device
            self.model = YOLO(model_path).to(self.device)
    
    def predict(self, image_path, conf=0.5):
        """ detect objects in image with conf threshold"""
        results = self.model.predict(image_path, conf=conf, device=self.device)
        return results
    
    def extract_boxes(self, results):
        """ return coordinates and classes of detected objects"""
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy().tolist()[0]
                cls = int(box.cls.cpu().numpy().tolist())
                conf = float(box.cls.cpu().numpy().tolist())
                detections.append({"bbox": xyxy, "class": cls, "conf": conf})
        return detections
                