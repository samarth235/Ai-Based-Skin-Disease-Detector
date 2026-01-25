🧠 DermAI Diagnostics

DermAI Diagnostics is a skin disease screening system that uses deep learning–based image analysis combined with basic clinical logic to assist in the preliminary identification of common skin diseases.
The system is designed as an AI-assisted decision support tool, not a replacement for professional medical diagnosis.

📌 Project Overview

Skin diseases often exhibit subtle visual patterns that are difficult to identify without expert knowledge.
This project leverages Convolutional Neural Networks (CNNs) to analyze skin images and predict the most likely disease class along with a confidence score.

The system integrates:

Deep learning–based image classification

Confidence-aware predictions

Differential diagnosis (Top-3 results)

Clinical recommendation logic

Explainable AI using Grad-CAM

A modern Streamlit-based user interface

🎯 Objectives

To automatically classify skin diseases from images

To assist early-stage screening using AI

To provide confidence-based predictions

To visualize model attention using explainable AI

To present results in a clinician-friendly format

🧪 Diseases Classified

The system is trained to classify the following skin conditions:

Acne

Eczema

Psoriasis

Ringworm

Melanoma

⚠️ Note: Predictions are limited to the diseases present in the training dataset.

🧠 Model Architecture
🔹 Backbone Network

Xception (Extreme Inception) CNN

Pre-trained on ImageNet

Fine-tuned for skin disease classification

🔹 Key Architectural Components

Input size: 299 × 299 × 3

Depthwise separable convolutions

Global Average Pooling (GAP)

Fully connected dense layers

Softmax output layer for probability distribution

⚙️ Working Pipeline

User uploads a skin image

Image is resized and preprocessed

Xception extracts deep visual features

Global Average Pooling compresses features

Dense layers perform classification

Softmax outputs class probabilities

Top-3 predictions are generated

Risk level and clinical suggestions are displayed

Grad-CAM highlights important image regions

📊 Output Provided

Predicted disease

Confidence score

Top-3 differential diagnosis

Risk assessment

Clinical recommendations

Explainable AI heatmap (Grad-CAM)

🖥️ User Interface

Built using Streamlit

Modern dark-themed medical UI

Step-by-step workflow:

Symptom questionnaire

Image upload

Diagnosis summary

Visual explanation

🧰 Tools & Technologies Used
Programming & Frameworks

Python

TensorFlow / Keras

Streamlit

NumPy

Pandas

PIL (Image Processing)

Deep Learning

Xception CNN

Transfer Learning

Softmax Classification

Grad-CAM (Explainable AI)
