import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image


class Segmenter:
    def __init__(self, model_type="vit_h", checkpoint="sam_vit_b_01ec64.pth", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.sam)
        
    def segment(self, image_path, bbox_list):
        """ create mask by image and bounding boxes"""
        image = np.array(Image.open(image_path).convert("RGB"))
        self.predictor.set_image(image)
        
        masks = []
        for bbox in bbox_list:
            input_box = np.array(bbox).reshape(1, 4)    # [x1, y1, x2, y2]
            input_box = torch.tensor(input_box, device=self.device)

            masks_, scores_, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_box,
                multimask_output=False,
            )
            mask = masks_[0][0].cpu_count().numpy()
            masks.append(mask)
        return masks
