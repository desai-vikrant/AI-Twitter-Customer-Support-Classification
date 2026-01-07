
import streamlit as st
import pickle
import numpy as np
import os
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Tweet Buddy 🐦",
    layout="wide",
    page_icon="🐦"
)

# -------------------------------------------------
# CSS: Background + Cartoon + Animations
# -------------------------------------------------
st.markdown("""
<style>
/* Background waves */
.stApp {
    background: linear-gradient(-45deg, #1DA1F2, #0f2027, #203a43, #1DA1F2);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main card */
.main-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    border-radius: 25px;
    padding: 3rem;
    max-width: 1100px;
    margin: auto;
    box-shadow: 0 25px 50px rgba(0,0,0,0.35);
}

/* Title */
.title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    color: white;
}

/* Bird container */
.bird-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 2rem 0;
}

/* Speech bubble */
.bubble {
    background: white;
    color: #333;
    padding: 1rem 1.5rem;
    border-radius: 20px;
    font-size: 1.1rem;
    margin-left: 20px;
    position: relative;
}

.bubble:after {
    content: "";
    position: absolute;
    left: -10px;
    top: 40%;
    border-width: 10px;
    border-style: solid;
    border-color: transparent white transparent transparent;
}

/* Floating animation */
.float {
    animation: floaty 3s ease-in-out infinite;
}

@keyframes floaty {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #1DA1F2, #0d8ddb);
    color: white;
    font-size: 1.2rem;
    padding: 0.7rem 2.5rem;
    border-radius: 40px;
    border: none;
}

.stButton button:hover {
    transform: scale(1.08);
}

/* Result box */
.result {
    margin-top: 2rem;
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.18);
    color: white;
    text-align: center;
    font-size: 1.3rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load model & tokenizer
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = load_model(os.path.join(BASE_DIR, "gru_model.h5"))
    tokenizer = pickle.load(open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb"))
    return model, tokenizer

model, tokenizer = load_artifacts()

# -------------------------------------------------
# Clean text
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">Tweet Buddy AI</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#e0f3ff;'>Type a tweet and let your AI bird decide if it’s a customer support query!</p>",
    unsafe_allow_html=True
)

# Cartoon Bird (SVG)
st.markdown("""
<div class="bird-container">
<svg class="float" width="120" height="120" viewBox="0 0 64 64">
  <circle cx="32" cy="32" r="28" fill="#1DA1F2"/>
  <circle cx="24" cy="26" r="4" fill="white"/>
  <circle cx="40" cy="26" r="4" fill="white"/>
  <circle cx="25" cy="27" r="2" fill="black"/>
  <circle cx="41" cy="27" r="2" fill="black"/>
  <polygon points="32,34 38,38 32,40" fill="#FFD700"/>
</svg>

<div class="bubble">
Hey! 👋<br>Type your tweet here 🐤
</div>
</div>
""", unsafe_allow_html=True)

user_text = st.text_area(
    "✍️ Your Tweet",
    height=150,
    placeholder="My order hasn’t arrived yet, can you help?"
)

col = st.columns([1,2,1])[1]
with col:
    predict = st.button("🚀 Analyze Tweet")

if predict:
    if user_text.strip() == "":
        st.warning("⚠️ Tweet Buddy needs some text!")
    else:
        clean = clean_text(user_text)
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=50, padding="post")
        prob = model.predict(pad, verbose=0)[0][0]

        label = (
            "📩 This tweet looks like a customer support request. Our team should respond to it."
            if prob >= 0.5
            else
            "📰 This tweet does not appear to be a customer support request and looks like a normal post."
            )


        st.markdown(
            f"""
            <div class="result">
            🧠 <b>Tweet Buddy Says:</b><br><br>
            {label}<br><br>
            Confidence: <b>{prob:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)


# #cd "D:\VS CODE\NexGen"
# # streamlit run app.py