import streamlit as st
from PIL import Image
import tempfile
import pandas as pd
import numpy as np

from predict import predict_skin_disease
from clinical_logic import confidence_badge, clinical_recommendation
from gradcam import make_gradcam_heatmap, overlay_heatmap

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="DermAI Diagnostics",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# STYLES
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.hero {
    text-align: center;
    margin: 40px 0 50px 0;
}

.hero h1 {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9ca3af;
    font-size: 18px;
    max-width: 720px;
    margin: auto;
}

.pill {
    padding: 10px 18px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    font-size: 14px;
    color: #cbd5f5;
}

.card {
    background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.9));
    border-radius: 18px;
    padding: 26px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
    margin-bottom: 30px;
}

.card h3 {
    font-size: 22px;
    margin-bottom: 18px;
}

.card p {
    color: #9ca3af;
    line-height: 1.6;
}

.badge-high { color: #ef4444; font-weight: 700; }
.badge-mid { color: #facc15; font-weight: 700; }
.badge-low { color: #22c55e; font-weight: 700; }

section[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    border-radius: 16px;
    padding: 30px;
    background: rgba(56,189,248,0.05);
}

img {
    border-radius: 12px;
}
.ai-explain {
    background: rgba(56, 189, 248, 0.08);
    border-left: 4px solid #38BDF8;
    padding: 16px;
    border-radius: 12px;
    color: #E5E7EB;
}
:root {
  --bg-main: #020617;
  --bg-card: #0F172A;
  --border-subtle: rgba(255,255,255,0.06);

  --text-main: #E5E7EB;
  --text-muted: #9CA3AF;

  --primary: #38BDF8;
  --secondary: #6366F1;

  --danger: #EF4444;
  --warning: #FACC15;
  --success: #22C55E;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <div style="font-size:60px;">🧠</div>
    <h1>DermaVision</h1>
    <p>
        AI-powered skin disease screening system
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:center; gap:14px; margin-bottom:40px;">
    <div class="pill">🧠 AI-Powered Analysis</div>
    <div class="pill">🩺 Clinical-Grade Accuracy</div>
    <div class="pill">⚡ Instant Results</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PATIENT QUESTIONNAIRE (FIXED)
# =====================================================
with st.form("symptom_form"):
    st.markdown("""
    <div class="card">
        <h3>📋 Patient Symptom Questionnaire</h3>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        itching = st.checkbox("Itching present?")
        duration = st.selectbox("Duration of condition", ["< 1 week", "1–4 weeks", "> 1 month"])

    with c2:
        pain = st.checkbox("Pain or tenderness?")
        location = st.selectbox("Affected body area", ["Face", "Scalp", "Body", "Hands / Feet"])

    submit = st.form_submit_button("Continue")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# IMAGE UPLOAD
# =====================================================
st.markdown("""
<div class="card">
    <h3>📤 Upload Skin Image</h3>
    <p>Upload a clear, well-lit image of the affected area.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =====================================================
# IMAGE PREVIEW (FIXED SIZE)
# =====================================================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.markdown("""
    <div class="card">
        <h3>🖼 Uploaded Image</h3>
    """, unsafe_allow_html=True)

    st.image(img, width=320)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ANALYSIS
# =====================================================
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(tmp.name)

    with st.spinner("Analyzing dermatological patterns…"):
        disease, confidence, risk, top3 = predict_skin_disease(tmp.name)

    badge = confidence_badge(confidence)
    recommendations = clinical_recommendation(disease)

    # ---------------- Diagnosis Summary ----------------
    st.markdown(f"""
    <div class="card">
        <h3>🩺 Diagnosis Summary</h3>
        <p style="font-size:22px;"><strong>{disease}</strong></p>
        <p>
            Confidence:
            <span style="font-weight:700; color:#38BDF8;">
                {confidence*100:.2f}%
            </span>
        </p>
        <p>Confidence Level: {badge}</p>
    </div>
    """, unsafe_allow_html=True)

    if "HIGH" in risk:
        st.error(risk)
    elif "MODERATE" in risk:
        st.warning(risk)
    else:
        st.success(risk)

    # ---------------- Clinical Recommendations (FIXED) ----------------
    st.markdown(f"""
    <div class="card">
        <h3>📋 Clinical Recommendations</h3>
        <p>
        Based on the detected condition <strong>{disease}</strong> and learned
        patterns from dermatological datasets, the AI suggests the following
        evidence-informed care actions:
        </p>
        <div style="margin-top:14px;">
        {''.join([f"<div class='ai-explain'>• {r}</div>" for r in recommendations])}
        </div>
        <p style="margin-top:14px; font-size:13px; opacity:0.75;">
        These recommendations are generated using clinical heuristics combined
        with deep learning feature interpretation.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ---------------- Top 3 ----------------
    df = pd.DataFrame(top3, columns=["Disease", "Probability"])
    df["Probability"] = df["Probability"] * 100

    st.markdown("""
    <div class="card">
        <h3>📊 Differential Diagnosis</h3>
    """, unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Grad-CAM ----------------
    model = load_model("models/skin_model_xception.h5", compile=False)

    arr = image.load_img(tmp.name, target_size=(299, 299))
    arr = image.img_to_array(arr)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    heatmap = make_gradcam_heatmap(arr, model, "block14_sepconv2_act")
    cam = overlay_heatmap(tmp.name, heatmap)

    st.markdown("""
    <div class="card">
        <h3>🧬 Explainable AI — Lesion Focus Map</h3>
        <p>Highlighted regions show where the model focused attention.</p>
    """, unsafe_allow_html=True)

    st.image(cam, width=360)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#9ca3af;">
⚠️ This tool is for educational and preliminary screening purposes only.<br>
It does NOT replace professional medical diagnosis.
</p>
""", unsafe_allow_html=True)
