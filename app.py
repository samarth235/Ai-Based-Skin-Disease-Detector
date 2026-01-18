import streamlit as st
from PIL import Image
import tempfile
from predict import predict_skin_disease
import pandas as pd
from gradcam import make_gradcam_heatmap, overlay_heatmap
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

st.set_page_config(page_title="Skin Disease AI", layout="centered")

st.title("🧠 AI-Based Skin Disease Detection")
st.write("Upload a skin image for preliminary AI-based screening.")

uploaded_file = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image", width=300)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:

        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        pil_image.save(tmp.name)

        disease, confidence, risk, top3 = predict_skin_disease(tmp.name)

    # ================= PREDICTION ==================

    st.subheader("Prediction Result")
    st.write(f"**Detected Condition:** {disease}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.warning(risk)

    # ================= TOP 3 ==================

    st.subheader("Top 3 Predictions")

    df = pd.DataFrame(top3, columns=["Disease", "Probability"])
    df["Probability"] = df["Probability"] * 100
    st.bar_chart(df.set_index("Disease"))

    # ================= GRAD-CAM ==================

    model = load_model("models/skin_model_xception.h5", compile=False)

    img_array = image.load_img(tmp.name, target_size=(299, 299))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    heatmap = make_gradcam_heatmap(img_array, model, "block14_sepconv2_act")
    cam_image = overlay_heatmap(tmp.name, heatmap)

    st.subheader("AI Attention Map (Grad-CAM)")
    st.image(cam_image, channels="BGR")

st.caption(
    "⚠ This tool is for educational & preliminary screening purposes only. "
    "It does NOT replace professional medical diagnosis."
)
