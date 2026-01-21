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

# ==================== STYLE ====================

st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.badge-high { color: #ff4b4b; font-weight: bold; }
.badge-mid { color: #facc15; font-weight: bold; }
.badge-low { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧠 AI-Powered Skin Disease Detection System")
st.markdown("##### Clinical Decision Support Tool")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a skin image",
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

    # ==================== DIAGNOSIS CARD ====================

    risk_class = "badge-low"
    if "HIGH" in risk:
        risk_class = "badge-high"
    elif "MODERATE" in risk:
        risk_class = "badge-mid"

    st.markdown(f"""
    <div class="card">
        <h3>🩺 Diagnosis Summary</h3>
        <p><b>Detected Condition:</b> {disease}</p>
        <p><b>Confidence:</b> {confidence*100:.2f}%</p>
        <p class="{risk_class}">{risk}</p>
    </div>
    """, unsafe_allow_html=True)

    # ==================== TOP 3 CARD ====================

    df = pd.DataFrame(top3, columns=["Disease", "Probability"])
    df["Probability"] = df["Probability"] * 100

    st.markdown("""
    <div class="card">
        <h3>📊 Differential Diagnosis (Top-3)</h3>
    </div>
    """, unsafe_allow_html=True)

    st.bar_chart(df.set_index("Disease"))

    # ==================== GRAD-CAM ====================

    model = load_model("models/skin_model_xception.h5", compile=False)

    img_array = image.load_img(tmp.name, target_size=(299, 299))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    heatmap = make_gradcam_heatmap(img_array, model, "block14_sepconv2_act")
    cam_image = overlay_heatmap(tmp.name, heatmap)

    st.markdown("""
    <div class="card">
        <h3>🧬 Explainable AI — Lesion Focus Map</h3>
    </div>
    """, unsafe_allow_html=True)

    st.image(cam_image, channels="BGR")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
🛑 **Medical Disclaimer**

This system is an AI-assisted screening tool only.  
It does **NOT replace professional dermatological diagnosis.**  
Always consult a certified dermatologist.
""")
