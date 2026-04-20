import os
import re
import pandas as pd
import joblib
import numpy as np

# Load trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')
DATA_FILE = os.path.join(os.path.dirname(__file__), 'mail_data.csv')

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 70)
print("  DATA VALIDATION & MISLABELED MESSAGE DETECTION")
print("=" * 70)


# ========== APPROACH 1: CONFIDENCE SCORE CHECK ==========
def check_confidence_mismatch(df, model, vectorizer):
    """Find messages where actual label doesn't match high-confidence prediction"""
    print("\n" + "-" * 70)
    print("APPROACH 1: CONFIDENCE SCORE CHECK")
    print("-" * 70)
    
    mismatches = []
    
    for idx, row in df.iterrows():
        text = row['text']
        actual_label = row['label']
        
        # Preprocess
        text_clean = text.lower()
        text_clean = re.sub(r'http\S+|www\S+', ' ', text_clean)
        text_clean = re.sub(r'[^\w\s@.-]', ' ', text_clean)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Vectorize and predict
        X_tfidf = vectorizer.transform([text_clean])
        pred = model.predict(X_tfidf)[0]
        proba = model.predict_proba(X_tfidf)[0]
        confidence = proba[int(pred)] * 100
        
        pred_label = "spam" if pred == 1 else "ham"
        
        # Flag if confidence > 90% and prediction doesn't match label
        if confidence > 90 and pred_label != actual_label:
            mismatches.append({
                'index': idx,
                'actual': actual_label,
                'predicted': pred_label,
                'confidence': confidence,
                'text': text[:80] + "..." if len(text) > 80 else text
            })
    
    if mismatches:
        print(f"\n✗ Found {len(mismatches)} HIGH-CONFIDENCE MISMATCHES (>90%):\n")
        for item in mismatches:
            print(f"  Line {item['index'] + 2}: [{item['actual'].upper()}] → Predicted: [{item['predicted'].upper()}] ({item['confidence']:.1f}%)")
            print(f"    Text: {item['text']}\n")
    else:
        print("\n✓ No high-confidence mismatches found!")
    
    return mismatches


# ========== APPROACH 2: KEYWORD SPAM DETECTION ==========
def check_keyword_indicators(df):
    """Find spam keywords in messages marked as ham"""
    print("\n" + "-" * 70)
    print("APPROACH 2: KEYWORD SPAM DETECTION")
    print("-" * 70)
    
    spam_keywords = [
        'free', 'winner', 'claim', 'prize', 'urgent', 'congratulations',
        'click here', 'call now', 'buy', 'offer', 'limited time', 'act now',
        'exclusive', 'guarantee', 'cash', 'money', 'winner', 'selected',
        'http', 'www', 'reply', 'text', 'txt', 'call', 'dial'
    ]
    
    suspicious_ham = []
    
    for idx, row in df.iterrows():
        if row['label'] == 'ham':  # Only check ham messages
            text = row['text'].lower()
            exclamation_count = text.count('!')
            
            # Check for keywords
            found_keywords = [kw for kw in spam_keywords if kw in text]
            
            # Flag if multiple spam keywords OR excessive exclamation marks
            if (len(found_keywords) >= 2 or exclamation_count >= 3):
                suspicious_ham.append({
                    'index': idx,
                    'keywords': found_keywords,
                    'exclamations': exclamation_count,
                    'text': text[:80] + "..." if len(text) > 80 else text
                })
    
    if suspicious_ham:
        print(f"\n⚠ Found {len(suspicious_ham)} HAM MESSAGES WITH SPAM INDICATORS:\n")
        for item in suspicious_ham:
            print(f"  Line {item['index'] + 2}: Keywords: {item['keywords']}, Exclamations: {item['exclamations']}")
            print(f"    Text: {item['text']}\n")
    else:
        print("\n✓ No suspicious ham messages found!")
    
    return suspicious_ham


# ========== COMBINED ANALYSIS ==========
def generate_correction_recommendations(df, confidence_mismatches, keyword_suspicious):
    """Generate recommendations for data correction"""
    print("\n" + "-" * 70)
    print("CORRECTION RECOMMENDATIONS")
    print("-" * 70)
    
    # Combine all suspicious indices
    suspicious_indices = set()
    for item in confidence_mismatches:
        suspicious_indices.add(item['index'])
    for item in keyword_suspicious:
        suspicious_indices.add(item['index'])
    
    if suspicious_indices:
        print(f"\n📋 Total suspicious messages found: {len(suspicious_indices)}")
        print("\nTo improve model accuracy, consider:")
        print("  1. Review these messages manually")
        print("  2. Change mislabeled messages from 'ham' to 'spam'")
        print("  3. Retrain the model with corrected data")
        print("  4. This will significantly improve spam detection accuracy")
    else:
        print("\n✓ Data appears clean! No corrections needed.")


# Run analysis
print("\n📊 Analyzing dataset for mislabeled messages...\n")
mismatches = check_confidence_mismatch(df, model, vectorizer)
suspicious = check_keyword_indicators(df)
recommendations = generate_correction_recommendations(df, mismatches, suspicious)

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70 + "\n")
