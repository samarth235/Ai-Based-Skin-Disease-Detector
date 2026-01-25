import os
import warnings
from PIL import Image, ImageFile

# Prevent native aborts
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

folders = [
    "dataset/train/acne",
    "dataset/train/eczema",
    "dataset/train/melanoma",
    "dataset/train/psoriasis",
    "dataset/train/ringworm",
]

bad = 0
total = 0

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        total += 1

        # Remove system junk
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(path)
            continue

        try:
            with Image.open(path) as img:
                img.verify()
        except:
            print("Removing corrupted:", path)
            os.remove(path)
            bad += 1

print(f"Scan complete: {total} files checked, {bad} removed.")



