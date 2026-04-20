import streamlit as st
import pandas as pd
import joblib
import re
import os
import pdfplumber
import io

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
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model_artifacts():
    # Ensure these files exist from your previous training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.joblib')
    vect_path = os.path.join(script_dir, 'vectorizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(vect_path):
        model = joblib.load(model_path)
        vect = joblib.load(vect_path)
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
    uploaded_file = st.file_uploader("Upload a .csv, .txt, or .pdf file", type=['csv', 'txt', 'pdf'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            # Find the text column automatically
            text_col = next((c for c in df.columns if c.lower() in ['text', 'message', 'body', 'sms']), None)
            
            if text_col:
                st.write(f"Found column: `{text_col}`. Classifying...")
                df['Cleaned'] = df[text_col].apply(simple_preprocess)
                features = vect.transform(df['Cleaned'])
                df['Prediction'] = model.predict(features)
                df['Label'] = df['Prediction'].map({0: 'HAM', 1: 'SPAM'})
                
                st.dataframe(df[[text_col, 'Label']], use_container_width=True)
                
                # Create CSV for download
                csv_data = df[[text_col, 'Label']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv_data,
                    file_name="spam_results.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                st.error("CSV must have a column named 'text', 'message', 'body', or 'sms'.")
        elif uploaded_file.name.endswith('.txt'):
            try:
                raw_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                raw_text = uploaded_file.read().decode("latin-1")
            
            st.write("Processing text file...")
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            
            if lines:
                results = []
                for i, line in enumerate(lines, 1):
                    clean_text = simple_preprocess(line)
                    tfidf_text = vect.transform([clean_text])
                    prediction = model.predict(tfidf_text)[0]
                    probability = model.predict_proba(tfidf_text)[0][prediction] * 100
                    label = 'SPAM' if prediction == 1 else 'HAM'
                    results.append({'Line': i, 'Message': line[:50] + '...' if len(line) > 50 else line, 'Label': label, 'Confidence': f"{probability:.1f}%"})
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Create CSV for download
                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv_data,
                    file_name="text_spam_results.csv",
                    mime="text/csv",
                    key="download_txt"
                )
            else:
                st.warning("No text found in file.")
        elif uploaded_file.name.endswith('.pdf'):
            st.write("Processing PDF file...")
            try:
                # Extract text from PDF
                pdf_text = ""
                with pdfplumber.open(uploaded_file) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                
                # Split into lines
                lines = [line.strip() for line in pdf_text.split('\n') if line.strip()]
                
                if lines:
                    st.write(f"📄 Extracted {len(lines)} lines from PDF")
                    
                    results = []
                    for i, line in enumerate(lines, 1):
                        clean_text = simple_preprocess(line)
                        tfidf_text = vect.transform([clean_text])
                        prediction = model.predict(tfidf_text)[0]
                        probability = model.predict_proba(tfidf_text)[0][prediction] * 100
                        label = 'SPAM' if prediction == 1 else 'HAM'
                        results.append({'Line': i, 'Message': line[:50] + '...' if len(line) > 50 else line, 'Label': label, 'Confidence': f"{probability:.1f}%"})
                    
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Show summary
                    spam_count = (df_results['Label'] == 'SPAM').sum()
                    ham_count = (df_results['Label'] == 'HAM').sum()
                    st.info(f"📊 Summary: {spam_count} SPAM, {ham_count} HAM out of {len(results)} messages")
                    
                    # Create CSV for download
                    csv_data = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_data,
                        file_name="pdf_spam_results.csv",
                        mime="text/csv",
                        key="download_pdf"
                    )
                else:
                    st.warning("No text could be extracted from the PDF.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")