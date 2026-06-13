import streamlit as st
import joblib
import pandas as pd

model = joblib.load('spam_model.joblib')

st.title("📧 Email Spam Detector")
st.write("Enter an email below to check if it's Spam or Ham.")

email_input = st.text_area("Paste your email text here:")

if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        features = pd.DataFrame([{"text": email_input}])
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is HAM (legitimate)!")