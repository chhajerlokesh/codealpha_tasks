# 🎙️ Speech Emotion Recognition (SER) using Deep Learning

An end-to-end Deep Learning project designed to recognize and classify human emotions from audio speech recordings. This project utilizes the **RAVDESS** dataset to train 1D Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. It also includes a fully interactive web application deployed using **Streamlit**.

## 🚀 Project Features
- **Comprehensive Feature Extraction:** Extracts MFCCs, Chroma, Mel Spectrogram, Zero-Crossing Rate (ZCR), and RMS Energy using `librosa`.
- **Deep Learning Architectures:** Compares the performance of spatial (1D CNN) and temporal (LSTM) models.
- **Interactive EDA:** Visualizes audio signals in both time-domain (waveforms) and frequency-domain (spectrograms).
- **Web App Interface:** An intuitive Streamlit dashboard to upload `.wav` or `.mp3` files, visualize confidence scores via Plotly, and predict emotions on the fly.

---

## 📂 Project Structure

```text
📁 speech-emotion-recognition
│
├── app.py                     # Streamlit web application for deployment
├── notebook.ipynb             # Jupyter Notebook for EDA, feature extraction, and model training
├── requirements.txt           # List of Python dependencies
├── best_emotion_model.keras   # Saved trained model (Generated after training)
├── scaler.joblib              # Fitted StandardScaler (Generated after training)
├── label_encoder.joblib       # Fitted LabelEncoder (Generated after training)
└── README.md                  # Project documentation