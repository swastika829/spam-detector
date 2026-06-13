import streamlit as st
import joblib
import pandas as pd

model = joblib.load('spam_model.joblib')

# Get the feature names the model was trained on
feature_names = model.feature_names_in_

st.title("📧 Email Spam Detector")
st.write("Enter an email below to check if it's Spam or Ham.")

email_input = st.text_area("Paste your email text here:")

if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Convert email text to word frequency format
        words = email_input.lower().split()
        input_row = {col: 0 for col in feature_names}
        for word in words:
            if word in input_row:
                input_row[word] += 1
        
        input_df = pd.DataFrame([input_row])
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is HAM (legitimate)!")
