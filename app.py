import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

st.markdown("""
    <style>
    .top-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; }
    .metric-row { display: flex; gap: 10px; margin-bottom: 1.5rem; }
    .metric-box { flex: 1; background: #f5f5f5; border-radius: 10px; padding: 12px 14px; }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 16px; font-weight: 600; color: #111; }
    .result-spam { background: #fff0f0; border: 1px solid #ffcccc; border-radius: 12px; padding: 16px 20px; margin-top: 1rem; }
    .result-ham { background: #f0fff4; border: 1px solid #b2f2bb; border-radius: 12px; padding: 16px 20px; margin-top: 1rem; }
    .result-title-spam { font-size: 18px; font-weight: 600; color: #cc0000; }
    .result-title-ham { font-size: 18px; font-weight: 600; color: #1a7a3a; }
    .prob-row { display: flex; gap: 10px; margin-top: 12px; }
    .prob-box { flex: 1; background: white; border-radius: 8px; padding: 10px 14px; border: 1px solid #eee; }
    .prob-label { font-size: 11px; color: #888; }
    .prob-val-spam { font-size: 22px; font-weight: 700; color: #cc0000; }
    .prob-val-ham { font-size: 22px; font-weight: 700; color: #1a7a3a; }
    .history-box { background: white; border: 1px solid #eee; border-radius: 12px; padding: 14px 18px; margin-top: 1.5rem; }
    .hist-item { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 1px solid #f0f0f0; }
    .hist-item:last-child { border-bottom: none; }
    .badge-spam { background: #fff0f0; color: #cc0000; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600; }
    .badge-ham { background: #f0fff4; color: #1a7a3a; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600; }
    .hist-text { font-size: 13px; color: #555; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px; }
    .hist-prob { font-size: 12px; color: #aaa; }
    .footer { text-align: center; color: #aaa; font-size: 12px; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('spam_model.joblib')
feature_names = model.feature_names_in_

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("## 📧 Email Spam Detector")
st.markdown("""
<div class="metric-row">
  <div class="metric-box"><div class="metric-label">Model</div><div class="metric-value">Logistic Regression</div></div>
  <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">~95%</div></div>
  <div class="metric-box"><div class="metric-label">Dataset</div><div class="metric-value">Kaggle BoW</div></div>
</div>
""", unsafe_allow_html=True)

email_input = st.text_area("📩 Paste your email text here:", height=150, placeholder="Type or paste an email message...")

if st.button("🔍 Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text first.")
    else:
        words = email_input.lower().split()
        input_row = {col: 0 for col in feature_names}
        for word in words:
            if word in input_row:
                input_row[word] += 1

        input_df = pd.DataFrame([input_row])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        spam_prob = round(proba[1] * 100)
        ham_prob = round(proba[0] * 100)

        label = "Spam" if prediction == 1 else "Ham"
        st.session_state.history.insert(0, {
            "label": label,
            "text": email_input[:60],
            "prob": spam_prob if prediction == 1 else ham_prob
        })
        st.session_state.history = st.session_state.history[:5]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-spam">
              <div class="result-title-spam">🚨 Spam detected</div>
              <div class="prob-row">
                <div class="prob-box"><div class="prob-label">Spam probability</div><div class="prob-val-spam">{spam_prob}%</div></div>
                <div class="prob-box"><div class="prob-label">Ham probability</div><div class="prob-val-ham">{ham_prob}%</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.progress(spam_prob)
        else:
            st.markdown(f"""
            <div class="result-ham">
              <div class="result-title-ham">✅ Ham — Legitimate email</div>
              <div class="prob-row">
                <div class="prob-box"><div class="prob-label">Ham probability</div><div class="prob-val-ham">{ham_prob}%</div></div>
                <div class="prob-box"><div class="prob-label">Spam probability</div><div class="prob-val-spam">{spam_prob}%</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.progress(ham_prob)

if st.session_state.history:
    items_html = ""
    for item in st.session_state.history:
        badge = f'<span class="badge-spam">Spam</span>' if item["label"] == "Spam" else f'<span class="badge-ham">Ham</span>'
        items_html += f'<div class="hist-item">{badge}<span class="hist-text">{item["text"]}...</span><span class="hist-prob">{item["prob"]}%</span></div>'
    st.markdown(f'<div class="history-box"><b style="font-size:13px;">Recent checks</b>{items_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Made by Swastika Kumari · B.Tech CSE · Galgotias University · 2025-26</div>', unsafe_allow_html=True)
