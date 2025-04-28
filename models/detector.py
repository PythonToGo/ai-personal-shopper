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
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # xyxy = box.xyxy.cpu().numpy().tolist()[0]
            conf = float(box.cls.cpu().item())
            cls = int(box.cls.cpu().item())
            
            detections.append({"bbox": [x1, y1, x2, y2], "class": cls, "conf": conf})
        return detections
                