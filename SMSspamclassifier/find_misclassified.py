import os
import re
import pandas as pd
import joblib

# Load trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')
DATA_FILE = os.path.join(os.path.dirname(__file__), 'mail_data.csv')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 80)
print("  FINDING MISCLASSIFIED MESSAGES IN TRAINING DATA")
print("=" * 80)

def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Check each message
misclassified = []
for idx, row in df.iterrows():
    actual_label = row['label'].lower()
    text = row['text']
    
    # Preprocess and predict
    text_clean = simple_preprocess(text)
    X_tfidf = vectorizer.transform([text_clean])
    pred = model.predict(X_tfidf)[0]
    confidence = model.predict_proba(X_tfidf)[0][pred] * 100
    
    pred_label = "spam" if pred == 1 else "ham"
    
    # Check if prediction doesn't match actual label
    if pred_label != actual_label:
        misclassified.append({
            'line': idx + 2,
            'actual': actual_label.upper(),
            'predicted': pred_label.upper(),
            'confidence': confidence,
            'text': text[:70] + "..." if len(text) > 70 else text
        })

if misclassified:
    print(f"\n❌ Found {len(misclassified)} MISCLASSIFIED MESSAGES:\n")
    for item in misclassified:
        print(f"Line {item['line']}: [{item['actual']}] → Predicted as [{item['predicted']}] ({item['confidence']:.1f}%)")
        print(f"   Text: {item['text']}\n")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS TO FIX:")
    print("=" * 80)
    print("\nOption 1: RETRAIN WITH CORRECTED LABELS")
    print("  - Edit mail_data.csv and correct the misclassified labels")
    print("  - Run: python retrain_model.py")
    print("  - This will rebuild the model with correct data")
    
    print("\nOption 2: USE A BETTER ALGORITHM")
    print("  - The model might benefit from:")
    print("    • Using SVM (Support Vector Machine)")
    print("    • Using Naive Bayes with better feature engineering")
    print("    • Ensemble methods (Random Forest, Gradient Boosting)")
    print("    • Deep Learning (Neural Networks)")
    
    print("\nOption 3: IMPROVE FEATURE ENGINEERING")
    print("  - Add custom spam indicators:")
    print("    • URL count")
    print("    • Exclamation mark count")
    print("    • ALL CAPS words count")
    print("    • Phone number presence")
    print("    • Money-related keywords (free, prize, cash, etc)")
    
else:
    print("\n✅ All messages correctly classified!")

print("\n" + "=" * 80 + "\n")
