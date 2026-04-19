import streamlit as st
import pandas as pd
import joblib
import re
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Spam Detector", page_icon="🛡️", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def simple_preprocess(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model_artifacts():
    # Ensure these files exist from your previous training script
    if os.path.exists('model.joblib') and os.path.exists('vectorizer.joblib'):
        model = joblib.load('model.joblib')
        vect = joblib.load('vectorizer.joblib')
        return model, vect
    return None, None

# --- UI LAYOUT ---
st.title("🛡️ Smart Spam Classifier")
st.write("Paste a message or upload a document to check for SPAM.")

model, vect = load_model_artifacts()

if model is None:
    st.error("❌ Model artifacts not found. Please run your training script first to generate 'model.joblib' and 'vectorizer.joblib'.")
    st.stop()

# --- INPUT SECTION ---
tabs = st.tabs(["💬 Paste Text", "📁 Upload Document"])

with tabs[0]:
    user_input = st.text_area("Enter your message here:", placeholder="e.g., You've won a $1000 gift card! Click here...")
    if st.button("Analyze Text"):
        if user_input.strip():
            clean_text = simple_preprocess(user_input)
            tfidf_text = vect.transform([clean_text])
            prediction = model.predict(tfidf_text)[0]
            probability = model.predict_proba(tfidf_text)[0][prediction] * 100

            if prediction == 1:
                st.error(f"🚨 **RESULT: SPAM** ({probability:.1f}% confidence)")
            else:
                st.success(f"✅ **RESULT: HAM (Clean)** ({probability:.1f}% confidence)")
        else:
            st.warning("Please enter some text first.")

with tabs[1]:
    uploaded_file = st.file_uploader("Upload a .csv or .txt file", type=['csv', 'txt'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Find the text column automatically
            text_col = next((c for c in df.columns if c.lower() in ['text', 'message', 'body']), None)
            
            if text_col:
                st.write(f"Found column: `{text_col}`. Classifying...")
                df['Cleaned'] = df[text_col].apply(simple_preprocess)
                features = vect.transform(df['Cleaned'])
                df['Prediction'] = model.predict(features)
                df['Label'] = df['Prediction'].map({0: 'HAM', 1: 'SPAM'})
                
                st.dataframe(df[[text_col, 'Label']], use_container_width=True)
                st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")
            else:
                st.error("CSV must have a column named 'text' or 'message'.")
        else:
            raw_text = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", raw_text, height=200)
            # Process as single block... (repeat logic above)