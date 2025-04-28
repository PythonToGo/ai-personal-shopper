from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_embedding(self, image_path):
        """ extract embedding vector from image"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    