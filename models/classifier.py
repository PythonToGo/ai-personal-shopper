from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

class ImageClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.labels = self.model.config.id2label
    
    def predict(self, image_path):
        """ return classification results for an cropped image"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = self.labels[predicted_class_idx]
            
        return predicted_label
