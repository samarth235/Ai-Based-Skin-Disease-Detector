import numpy as np
from disease_mapper import refine_diagnosis
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

MODEL_PATH = "models/skin_model_xception.h5"
CLASS_PATH = "models/class_indices.npy"
IMG_SIZE = 299

model = load_model(MODEL_PATH, compile=False)

# Load correct class order
class_indices = np.load(CLASS_PATH, allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}

def predict_skin_disease(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)[0]

    # Temperature scaling
    temperature = 0.6
    preds = np.log(preds + 1e-9) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))

    # Top-3 predictions
    top_idx = preds.argsort()[-3:][::-1]
    top3 = [(labels[i], float(preds[i])) for i in top_idx]

    best_label, best_conf = top3[0]

    if best_conf < 0.40:
        return "Uncertain", best_conf, "Image unclear — try another photo", top3

    if best_label.lower() == "melanoma":
        risk = "HIGH RISK — Consult dermatologist immediately"
    elif best_conf > 0.75:
        risk = "MODERATE RISK — Medical advice recommended"
    else:
        risk = "LOW RISK — Monitor condition"

    detailed_name, detail_note = refine_diagnosis(best_label.capitalize(), best_conf)

    final_label = f"{detailed_name}"

    return final_label, best_conf, risk + f" ({detail_note})", top3
