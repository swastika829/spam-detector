import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .title { text-align: center; font-size: 2.5em; font-weight: bold; color: #1a1a2e; margin-bottom: 0; }
    .subtitle { text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1em; }
    .result-spam { background-color: #ff4b4b; color: white; padding: 20px; border-radius: 12px; text-align: center; font-size: 1.4em; font-weight: bold; }
    .result-ham { background-color: #21c55d; color: white; padding: 20px; border-radius: 12px; text-align: center; font-size: 1.4em; font-weight: bold; }
    .stTextArea textarea { border-radius: 12px; font-size: 1em; }
    .stButton > button { width: 100%; background-color: #1a1a2e; color: white; border-radius: 12px; padding: 12px; font-size: 1.1em; border: none; }
    .stButton > button:hover { background-color: #16213e; }
    .footer { text-align: center; color: #aaa; margin-top: 40px; font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('spam_model.joblib')
feature_names = model.feature_names_in_

# Header
st.markdown('<div class="title">📧 Email Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning · Logistic Regression · 95% Accuracy</div>', unsafe_allow_html=True)

st.divider()

# Stats row
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Logistic Regression")
col2.metric("Accuracy", "~95%")
col3.metric("Dataset", "Kaggle BoW")

st.divider()

# Input
email_input = st.text_area("📩 Paste your email text here:", height=180, placeholder="Type or paste an email message...")

if st.button("🔍 Check Email"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text first.")
    else:
        words = email_input.lower().split()
        input_row = {col: 0 for col in feature_names}
        for word in words:
            if word in input_row:
                input_row[word] += 1

        input_df = pd.DataFrame([input_row])
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0]
        spam_prob = round(confidence[1] * 100, 2)
        ham_prob = round(confidence[0] * 100, 2)

        st.divider()
        if prediction == 1:
            st.markdown('<div class="result-spam">🚨 SPAM DETECTED!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">✅ HAM — Legitimate Email</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Confidence Scores")
        col1, col2 = st.columns(2)
        col1.metric("✅ HAM Probability", f"{ham_prob}%")
        col2.metric("🚨 SPAM Probability", f"{spam_prob}%")

        st.progress(int(spam_prob))

# Footer
st.markdown('<div class="footer">Made by Swastika Kumari · B.Tech CSE · Galgotias University · 2025-26</div>', unsafe_allow_html=True)
