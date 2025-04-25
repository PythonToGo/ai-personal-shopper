# setup_structure.py
import os

folders = [
    "data/raw", "data/processed", "data/embeddings", "data/faiss_index",
    "models", "utils", "interface"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

with open("README.md", "w") as f:
    f.write("# AI Personal Shopper\n\nEnd-to-end fashion recommendation system.")
print("Created README.md")

with open("main.py", "w") as f:
    f.write("# Entry point for AI Personal Shopper pipeline\n")
print("Created main.py")
