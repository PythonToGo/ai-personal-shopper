# main.py
from models.detector import ObjectDetector
from models.classifier import ImageClassifier
from models.segmenter import Segmenter
from models.embedder import ImageEmbedder
from models.recommender import FaissRecommender
import os
from PIL import Image
import numpy as np
import cv2

def run_detection_and_segmentation(image_path):
    detector = ObjectDetector(model_path="yolov8n.pt")
    classifier = ImageClassifier(model_name="google/vit-base-patch16-224")
    segmenter = Segmenter(model_type="vit_b", checkpoint_path="data/models/sam_vit_b_01ec64.pth")

    results = detector.predict(image_path, conf=0.5)
    detections = detector.extract_boxes(results)
    print("Detections: ", detections)

    os.makedirs("data/processed/crops", exist_ok=True)
    image = Image.open(image_path)

    bbox_list = []
    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        bbox_list.append(bbox)

        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = f"data/processed/crops/crop_{idx}.jpg"
        crop.save(crop_path)

        label = classifier.predict(crop_path)
        print(f"Crop {idx}: Predicted label = {label}")

    # Segment
    masks = segmenter.segment(image_path, bbox_list)
    print("Masks: ", masks)

    os.makedirs("data/processed/masks", exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"data/processed/masks/mask_{idx}.jpg", mask_img)
        print(f"Saved mask {idx}")

def run_embedding_and_recommendation():
    embedder = ImageEmbedder(model_name="openai/clip-vit-base-patch32")
    recommender = FaissRecommender(dim=512, index_path="data/faiss_index/index.faiss")

    crop_dir = "data/processed/crops"
    image_paths = [os.path.join(crop_dir, f) for f in os.listdir(crop_dir) if f.endswith(".jpg")]
    image_paths.sort()
    
    if len(image_paths) == 0:
        print("No crops found! Exiting.")
        return
    for img_path in image_paths:
        emb = embedder.get_embedding(img_path)
        recommender.add(emb, meta=img_path)

    if recommender.index.ntotal > 0:
        os.makedirs(os.path.dirname(recommender.index_path), exist_ok=True)
        recommender.save()
        print("Faiss saved")
    else:
        print("No embeddings added.")

    # search test
    if len(image_paths) >= 1:
        query_emb = embedder.get_embedding(image_paths[0])
        results = recommender.search(query_emb, topk=min(2, len(image_paths)))

    print("Top matches:")
    for meta, dist in results:
        print(f"Match: {meta}, Distance: {dist}")

if __name__ == "__main__":
    image_path = "data/raw/ober.jpg"
    run_detection_and_segmentation(image_path)
    run_embedding_and_recommendation()
