import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import plotly.express as px
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

# ---------------------------------------------------------
# 1. Model & Artifact Loading (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('best_emotion_model.keras')
        scaler = joblib.load('scaler.joblib')
        le = joblib.load('label_encoder.joblib')
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.stop()

model, scaler, le = load_artifacts()

# ---------------------------------------------------------
# 2. Feature Extraction Pipeline
# ---------------------------------------------------------
def extract_features_sequence(file_upload, max_len=150, sr=22050):
    """
    Replicates the exact extraction pipeline from the Jupyter Notebook.
    """
    try:
        # Load audio directly from UploadedFile object
        y, sample_rate = librosa.load(file_upload, sr=sr)
        y, _ = librosa.effects.trim(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40).T
        chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate).T
        mel = librosa.feature.melspectrogram(y=y, sr=sample_rate).T
        zcr = librosa.feature.zero_crossing_rate(y=y).T
        rms = librosa.feature.rms(y=y).T
        
        features = np.hstack([mfcc, chroma, mel, zcr, rms])
        
        if features.shape[0] < max_len:
            pad_width = max_len - features.shape[0]
            features = np.pad(features, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_len, :]
            
        return features, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# 3. UI Design & Audio Input Interface
# ---------------------------------------------------------
st.title("🎙️ Emotion Recognition from Speech")
st.markdown("""
Upload an audio clip, and our Deep Learning model will analyze the acoustic features to determine the speaker's emotional state.
""")

st.sidebar.header("Instructions")
st.sidebar.info("""
1. Upload a `.wav` or `.mp3` file containing human speech.
2. Play the audio to verify it's the correct clip.
3. Click **Analyze Emotion** to run the prediction.
""")

uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Inline Audio Player
    st.audio(uploaded_file, format='audio/wav')
    
    # ---------------------------------------------------------
    # 4. Prediction & Analytics
    # ---------------------------------------------------------
    if st.button("Analyze Emotion", type="primary"):
        with st.spinner("Extracting spectral features and predicting..."):
            
            # Extract
            features, err = extract_features_sequence(uploaded_file)
            
            if err:
                st.error(f"Error processing audio: {err}. Please ensure the file is not corrupted.")
            else:
                # Scale
                time_steps, num_features = features.shape
                features_scaled = scaler.transform(features) # Scaler expects 2D
                features_input = np.expand_dims(features_scaled, axis=0) # Reshape to (1, time_steps, features)
                
                # Predict
                predictions = model.predict(features_input)[0]
                pred_index = np.argmax(predictions)
                predicted_emotion = le.classes_[pred_index].capitalize()
                confidence = predictions[pred_index] * 100
                
                st.divider()
                
                # Layout for Results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Result")
                    # Visual Formatting Badge
                    st.success(f"**Predicted Emotion:** \n### {predicted_emotion}")
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
                with col2:
                    st.subheader("Emotion Probabilities")
                    
                    # Create a DataFrame for Plotly
                    prob_df = pd.DataFrame({
                        'Emotion': [c.capitalize() for c in le.classes_],
                        'Probability (%)': predictions * 100
                    })
                    prob_df = prob_df.sort_values(by='Probability (%)', ascending=True)
                    
                    # Interactive Bar Chart
                    fig = px.bar(
                        prob_df, 
                        x='Probability (%)', 
                        y='Emotion', 
                        orientation='h',
                        color='Probability (%)',
                        color_continuous_scale='Blues',
                        text_auto='.1f'
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title="Confidence Probability (%)",
                        yaxis_title="",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)