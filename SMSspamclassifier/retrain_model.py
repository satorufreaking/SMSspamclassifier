import os
import re
import io
import zipfile
import joblib
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_FILE = os.path.join(os.path.dirname(__file__), "mail_data.csv")

print("\n" + "=" * 70)
print("  RETRAINING MODEL WITH CORRECTED DATA")
print("=" * 70)

def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s@.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset
df = pd.read_csv(DATA_FILE).copy()

# CORRECTION: Fix known mislabeled messages
# Based on validation analysis, change these from their current labels:
corrections = {
    752: 'ham',    # Line 753: Actually not spam, personal message
    1431: 'ham',   # Line 1432: For sale item, not spam
    3361: 'ham',   # Line 3362: Personal call message
    5459: 'ham'    # Line 5460: Shopping break offer - could be borderline
}

corrected_count = 0
for idx, new_label in corrections.items():
    if idx < len(df) and df.iloc[idx]['label'] != new_label:
        old_label = df.iloc[idx]['label']
        df.at[idx, 'label'] = new_label
        print(f"\n✓ Corrected Line {idx + 2}: '{old_label}' → '{new_label}'")
        print(f"  Text: {df.iloc[idx]['text'][:60]}...")
        corrected_count += 1

print(f"\n\nTotal corrections applied: {corrected_count}")

# Map labels to binary
labels = df['label'].unique().tolist()
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

df['text_clean'] = df['text'].apply(simple_preprocess)

X = df['text_clean']
y = df['label_num']

print("\n" + "-" * 70)
print("TRAINING IMPROVED MODEL")
print("-" * 70)

print(f"\nDataset split:")
print(f"  Training messages: {int(len(df) * 0.8)}")
print(f"  Testing messages: {int(len(df) * 0.2)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create vectorizer and train
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining Logistic Regression model...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Evaluate
print("\nEvaluating model accuracy...")
preds = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, preds)

print(f"\n✓ Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()

print("\nDetailed Results:")
print(f"  ✓ True Positives (Spam correctly detected): {tp}")
print(f"  ✓ True Negatives (Ham correctly detected): {tn}")
print(f"  ✗ False Negatives (Spam missed): {fn}")
print(f"  ✗ False Positives (Ham marked as spam): {fp}")

# Save improved model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
vect_path = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vect_path)

print("\n" + "-" * 70)
print("✓ MODEL SAVED SUCCESSFULLY!")
print("-" * 70)
print(f"  Model: {model_path}")
print(f"  Vectorizer: {vect_path}")
print("\nYour Streamlit app will now use the improved model!")
print("=" * 70 + "\n")
