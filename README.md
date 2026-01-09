# SMSspamclassifier
import os
import re
import io
import zipfile
import joblib
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = os.path.join(os.path.dirname(__file__), "mail_data.csv")


def download_sms_dataset(target_path: str) -> bool:
	"""Download the SMS Spam Collection from UCI and save as a CSV with columns (label,text).
	Returns True on success, False otherwise."""
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
	print(f"Downloading dataset from {url} ...")
	try:
		resp = requests.get(url, timeout=30)
		resp.raise_for_status()
		with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
			# file inside zip is 'SMSSpamCollection'
			name = "SMSSpamCollection"
			if name in z.namelist():
				with z.open(name) as f:
					raw = f.read().decode("utf-8", errors="ignore")
					rows = [r.split('\t', 1) for r in raw.splitlines() if r.strip()]
					df = pd.DataFrame(rows, columns=["label", "text"])
					df.to_csv(target_path, index=False)
					print(f"Saved dataset to {target_path}")
					return True
			else:
				print("Expected file not found inside the zip archive.")
				return False
	except Exception as e:
		print("Failed to download dataset:", e)
		return False


def load_dataset(path: str) -> pd.DataFrame:
	# Try to read common formats; fall back to the UCI format.
	if not os.path.exists(path):
		raise FileNotFoundError(path)

	try:
		df = pd.read_csv(path)
		# If it has no obvious text column, try tsv/no-header
		if not any(c.lower() in ("text", "message", "body") for c in df.columns):
			raise ValueError("No text column found")
		# Normalize column names
		cols = {c: c for c in df.columns}
		for c in df.columns:
			lc = c.lower()
			if lc in ("label", "class", "spam/ham"):
				cols[c] = "label"
			if lc in ("text", "message", "body"):
				cols[c] = "text"
		df = df.rename(columns=cols)
		return df[["label", "text"]]
	except Exception:
		# Try reading as the UCI SMSSpamCollection format (tab-separated, no header)
		df = pd.read_csv(path, sep='\t', header=None, names=["label", "text"], encoding='utf-8', engine='python')
		return df


def simple_preprocess(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = text.lower()
	# remove urls
	text = re.sub(r'http\S+|www\S+', ' ', text)
	# remove non-word characters (keep basic punctuation)
	text = re.sub(r'[^\w\s@.-]', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text


def build_and_eval(df: pd.DataFrame):
	# Map labels to binary
	labels = df['label'].unique().tolist()
	if set(['ham','spam']).issubset(set(labels)):
		df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
	else:
		# try numeric
		try:
			df['label_num'] = df['label'].astype(int)
		except Exception:
			# fallback: treat the first unique as 0 else 1
			mapping = {labels[0]: 0}
			for v in labels[1:]:
				mapping[v] = 1
			df['label_num'] = df['label'].map(mapping)

	df['text_clean'] = df['text'].apply(simple_preprocess)

	X = df['text_clean']
	y = df['label_num']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

	vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1,2))
	X_train_tfidf = vectorizer.fit_transform(X_train)
	X_test_tfidf = vectorizer.transform(X_test)

	clf = LogisticRegression(max_iter=1000)
	clf.fit(X_train_tfidf, y_train)

	preds = clf.predict(X_test_tfidf)
	probs = clf.predict_proba(X_test_tfidf)[:, 1]

	print("Accuracy:", accuracy_score(y_test, preds))
	print("\nClassification report:\n", classification_report(y_test, preds))
	print("Confusion matrix:\n", confusion_matrix(y_test, preds))

	# Save artifacts
	joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'model.joblib'))
	joblib.dump(vectorizer, os.path.join(os.path.dirname(__file__), 'vectorizer.joblib'))
	print("Saved `model.joblib` and `vectorizer.joblib` in script directory.")

	return clf, vectorizer


if __name__ == '__main__':
	# Ensure dataset exists or download a default SMS spam dataset
	if not os.path.exists(DATA_FILE):
		print(f"Dataset not found at {DATA_FILE}.")
		ok = download_sms_dataset(DATA_FILE)
		if not ok:
			print("Could not download a default dataset. Please place a CSV at:", DATA_FILE)
			raise SystemExit(1)

	df = load_dataset(DATA_FILE)
	print(f"Dataset loaded: {len(df)} rows. Columns: {list(df.columns)}")

	clf, vect = build_and_eval(df)

	# quick example
	examples = [
		"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
		"Hey, are we still meeting for lunch today?"
	]
	ex_clean = [simple_preprocess(t) for t in examples]
	ex_tfidf = vect.transform(ex_clean)
	preds = clf.predict(ex_tfidf)
	print('\nSample predictions:')
	for t, p in zip(examples, preds):
		label = 'spam' if int(p) == 1 else 'ham'
		print(f" - [{label}] {t}")
