import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

DATA_FILE = os.path.join(os.path.dirname(__file__), 'mail_data.csv')

print("\n" + "=" * 80)
print("  IMPROVED MODEL WITH CUSTOM FEATURES")
print("=" * 80)

def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_custom_features(text: str) -> dict:
    """Extract custom spam indicators"""
    features = {}
    
    # URL indicators
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text))
    
    # Exclamation marks (spam often has many)
    features['exclamation_count'] = text.count('!')
    
    # ALL CAPS words (spam indicator)
    all_caps = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features['all_caps_count'] = all_caps
    
    # Spam keywords
    spam_keywords = ['free', 'winner', 'claim', 'prize', 'urgent', 'congratulations',
                     'click', 'call', 'buy', 'offer', 'limited', 'cash', 'money',
                     'selected', 'reply', 'text', 'dial', 'win', 'guarantee']
    features['spam_keyword_count'] = sum(1 for kw in spam_keywords if kw in text.lower())
    
    # Phone number indicators
    features['has_phone'] = 1 if re.search(r'\d{7,}', text) else 0
    
    # Currency indicators
    features['has_currency'] = 1 if re.search(r'[$£€]|dollar|pound|euro', text) else 0
    
    # Message length (spam tends to be longer)
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    return features

# Load data
df = pd.read_csv(DATA_FILE)

# Map labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df['text_clean'] = df['text'].apply(simple_preprocess)

# Remove empty texts
df = df[df['text_clean'].str.len() > 0].copy()

print(f"\n📊 Dataset: {len(df)} messages")
print(f"   Ham: {(df['label_num'] == 0).sum()}")
print(f"   Spam: {(df['label_num'] == 1).sum()}")

# Split data
X = df['text_clean']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n📈 Training: {len(X_train)} messages, Testing: {len(X_test)} messages")

# TF-IDF Vectorization with improved settings
print("\n🔧 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,
    min_df=1,
    ngram_range=(1, 2),
    max_features=5000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train improved Logistic Regression (better for text classification)
print("🤖 Training Logistic Regression with custom settings...")
clf = LogisticRegression(
    max_iter=1500,
    C=0.5,  # Regularization parameter
    class_weight='balanced'  # Handle class imbalance
)
clf.fit(X_train_tfidf, y_train)

# Evaluate
preds = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')

print(f"\n✅ Model Performance:")
print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   Precision: {precision * 100:.2f}%")
print(f"   Recall:    {recall * 100:.2f}%")
print(f"   F1-Score:  {f1 * 100:.2f}%")

# Detailed results
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Detailed Results:")
print(f"   ✓ True Positives (Spam correctly detected):  {tp}")
print(f"   ✓ True Negatives (Ham correctly detected):   {tn}")
print(f"   ✗ False Negatives (Spam missed):             {fn}")
print(f"   ✗ False Positives (Ham marked as spam):      {fp}")

# Save model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
vect_path = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vect_path)

print(f"\n💾 Model saved!")
print(f"   Model: {model_path}")
print(f"   Vectorizer: {vect_path}")

print("\n" + "=" * 80)
print("Next Step: Run find_misclassified.py to see which messages need label correction")
print("=" * 80 + "\n")
