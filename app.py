#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pyttsx3

# ---------------------------
# Load the trained model
# ---------------------------
model = load_model("sign_language_model.h5")

# ---------------------------
# A–Z Labels
# ---------------------------
labels = [chr(i) for i in range(65, 91)]
image_size = 64

# ---------------------------
# TTS Engine for Upload Prediction
# ---------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Sign Language Recognizer", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🖐️ Sign Language Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image or use webcam to recognize hand sign letters (A–Z).</p>", unsafe_allow_html=True)

# ---------------------------
# Sidebar with Instructions
# ---------------------------
with st.sidebar:
    st.header("📌 Instructions")
    st.markdown("""
    - Upload a hand gesture image (clear & centered).
    - For webcam, place your hand inside the white box.
    - Voice output works only with uploaded images.
    """)
    tts_enabled = st.toggle("🔊 Enable Voice Output (Upload Only)", value=True)

# ---------------------------
# Upload Mode
# ---------------------------
st.subheader("📤 Upload Hand Sign Image")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((image_size, image_size))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, image_size, image_size, 1)

    result = model.predict(img_array)
    prediction = labels[np.argmax(result)]
    confidence = np.max(result)

    st.image(uploaded_file, caption="Uploaded Image", width=250)
    st.markdown(f"""
    <div style='padding: 10px; background-color: #e0f7fa; border-radius: 10px; text-align: center; color: black;'>
        <h3>✅ Prediction: <span style='color: #00796b;'>{prediction}</span></h3>
        <p>Confidence: <b>{confidence:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if tts_enabled and st.button("🔊 Speak Prediction", key="speak_upload"):
        speak(f"The predicted letter is {prediction}")

# ---------------------------
# Webcam Mode
# ---------------------------
st.subheader("📷 Use Webcam for Live Prediction")

start_webcam = st.checkbox("📸 Start Webcam")

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
    st.session_state.latest_confidence = None

if start_webcam:
    st.warning("Uncheck to stop webcam.")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while start_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        roi = frame[40:400, 0:300]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (image_size, image_size))
        normalized = resized.astype("float32") / 255.0
        reshaped = normalized.reshape(1, image_size, image_size, 1)

        result = model.predict(reshaped)
        prediction = labels[np.argmax(result)]
        confidence = np.max(result)

        st.session_state.latest_prediction = prediction
        st.session_state.latest_confidence = confidence

        # Draw ROI & prediction
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        cv2.putText(frame, f"Live: {prediction} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

        # Stop webcam if checkbox unchecked (loop ends)
        if not st.session_state.get("start_webcam_state", True):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Latest Prediction Display (Post-Webcam)
# ---------------------------
if st.session_state.latest_prediction:
    st.markdown(f"""
    <div style='padding: 10px; background-color: #f9fbe7; border-radius: 10px; text-align: center; color: black;'>
        <h4>📷 Latest Webcam Prediction: <span style='color: #33691e;'>{st.session_state.latest_prediction}</span></h4>
        <p>Confidence: <b>{st.session_state.latest_confidence:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("🎓 Final Year Project ")


# In[ ]:





# In[ ]:




