# 🧠 DermAI Diagnostics

**Skin Disease Screening System using Deep Learning**

DermAI Diagnostics is a **computer vision–based skin disease screening system** that classifies dermatological conditions from images using a **transfer-learned CNN model**, confidence scoring, and explainable AI.

> Designed as an **academic + engineering project** demonstrating applied deep learning in healthcare.

---

## 🔍 Problem Statement

Early identification of skin diseases is challenging due to:

* Visual similarity between conditions
* Limited access to dermatologists
* Subjective manual diagnosis

This project aims to provide a **preliminary AI-assisted screening tool** using image-based analysis.

---

## 💡 Key Highlights (For Resume)

* Implemented **Xception CNN** using transfer learning
* Achieved **confidence-based multi-class classification**
* Integrated **Grad-CAM** for explainable AI visualization
* Built a **clinical decision logic layer** on top of model predictions
* Developed a **modern Streamlit UI** for end-to-end usability

---

## 🧪 Diseases Classified

* Acne
* Eczema
* Psoriasis
* Ringworm
* Melanoma

---

## 🧠 Model Architecture

* **Input:** 299 × 299 RGB skin image
* **Backbone:** Xception (pre-trained on ImageNet)
* **Feature Compression:** Global Average Pooling
* **Classifier:** Fully Connected Dense Layers
* **Output:** Softmax probability distribution

---

## ⚙️ Tech Stack

* **Programming:** Python
* **Deep Learning:** TensorFlow, Keras
* **UI:** Streamlit
* **Explainability:** Grad-CAM
* **Data Handling:** NumPy, Pandas
* **Image Processing:** PIL

---

## 📂 Project Structure

```
skin-disease-ai/
├── app.py                  # Streamlit frontend
├── predict.py              # Model inference logic
├── gradcam.py              # Explainable AI
├── clinical_logic.py       # Risk & recommendation logic
├── models/
│   ├── skin_model_xception.h5
│   └── class_indices.npy
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Output Provided

* Predicted disease class
* Confidence score
* Risk categorization (Low / Moderate / High)
* Top-3 probable conditions
* Visual attention map (Grad-CAM)

---

## ❌ Limitations

* Limited number of disease classes
* Model performance depends on image quality
* Not a substitute for professional diagnosis

---

## 🔮 Future Scope

* Training on larger datasets (ISIC / HAM10000)
* Mobile / web deployment
* PDF medical report generation
* Multi-image patient history analysis

---

## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.
It does **not** replace certified medical diagnosis.

---
