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
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_FILE = os.path.join(os.path.dirname(__file__), "mail_data.csv")


def print_header(title):
	"""Print a simple, easy-to-read header"""
	print("\n" + "=" * 60)
	print(f"  {title}")
	print("=" * 60)


def print_simple_message(message):
	"""Print a message in simple, clear language"""
	print(f"\n>>> {message}")


def print_result(label, value, explanation=""):
	"""Print a result with label and simple explanation"""
	print(f"\n  {label}: {value}")
	if explanation:
		print(f"  What does this mean? {explanation}")


def download_sms_dataset(target_path: str) -> bool:
	"""Download the SMS Spam Collection from UCI and save as a CSV with columns (label,text).
	Returns True on success, False otherwise."""
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
	print_simple_message("Getting SMS messages from the internet...")
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
					print_simple_message(f"Saved the messages to: {target_path}")
					return True
			else:
				print_simple_message("Could not find the messages file inside the download.")
				return False
	except Exception as e:
		print_simple_message(f"Could not download. Error: {e}")
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
	text = re.sub(r'http\S+|www\S+', 'URL', text)
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

	# Remove null/empty texts
	df = df[df['text_clean'].str.len() > 0].copy()

	X = df['text_clean']
	y = df['label_num']

	print_header("TEACHING THE COMPUTER")
	print_simple_message(f"Splitting messages: {int(len(df) * 0.8)} for learning, {int(len(df) * 0.2)} for testing")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

	vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=1, ngram_range=(1,2), max_features=5000)
	X_train_tfidf = vectorizer.fit_transform(X_train)
	X_test_tfidf = vectorizer.transform(X_test)

	print_simple_message("Converting messages to numbers (computer language)...")

	clf = LogisticRegression(max_iter=1000)
	print_simple_message("Teaching the computer to recognize SPAM...")
	clf.fit(X_train_tfidf, y_train)

	print_header("TESTING THE COMPUTER")
	print_simple_message("Now testing if the computer learned correctly...")

	preds = clf.predict(X_test_tfidf)
	probs = clf.predict_proba(X_test_tfidf)[:, 1]

	accuracy = accuracy_score(y_test, preds)
	print_result("SUCCESS RATE", f"{accuracy * 100:.1f}%", 
	             "Out of 100 messages, the computer correctly identified this many")

	# Confusion matrix
	cm = confusion_matrix(y_test, preds)
	tn, fp, fn, tp = cm.ravel()

	print("\n" + "-" * 60)
	print("  DETAILED RESULTS:")
	print("-" * 60)
	print(f"\n  Real SPAM messages it correctly found:    {tp}")
	print(f"  Real HAM messages it correctly found:     {tn}")
	print(f"  Real SPAM it missed (thought was HAM):    {fn}")
	print(f"  Real HAM it wrongly marked as SPAM:       {fp}")

	# Simple interpretation
	if accuracy > 0.95:
		print("\n  >>> The computer is EXCELLENT at finding spam! Very accurate!")
	elif accuracy > 0.85:
		print("\n  >>> The computer is GOOD at finding spam! Mostly accurate!")
	elif accuracy > 0.70:
		print("\n  >>> The computer is OK at finding spam. Could be better.")
	else:
		print("\n  >>> The computer needs more practice at finding spam.")

	# Save artifacts
	joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'model.joblib'))
	joblib.dump(vectorizer, os.path.join(os.path.dirname(__file__), 'vectorizer.joblib'))
	print("\n  Saved the learned model for future use.")

	return clf, vectorizer


if __name__ == '__main__':
	print_header("SMS SPAM CLASSIFIER - SIMPLE VERSION")
	print_simple_message("Checking for SMS messages file...")

	# Ensure dataset exists or download a default SMS spam dataset
	if not os.path.exists(DATA_FILE):
		print_simple_message(f"File not found at: {DATA_FILE}")
		ok = download_sms_dataset(DATA_FILE)
		if not ok:
			print_simple_message("Could not get the messages. Please place a CSV file at: " + DATA_FILE)
			raise SystemExit(1)

	df = load_dataset(DATA_FILE)
	print_header("DATASET LOADED")
	print_simple_message(f"We have {len(df)} SMS messages to learn from!")
	print_simple_message("These messages are labeled SPAM or HAM (not spam)")

	clf, vect = build_and_eval(df)

	# quick example
	print_header("TESTING WITH SAMPLE MESSAGES")
	print_simple_message("Let's test the computer with some real messages:\n")

	examples = [
		"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
		"Hey, are we still meeting for lunch today?"
	]

	for i, example in enumerate(examples, 1):
		ex_clean = simple_preprocess(example)
		ex_tfidf = vect.transform([ex_clean])
		pred = clf.predict(ex_tfidf)[0]
		prob = clf.predict_proba(ex_tfidf)[0]

		label = "SPAM" if int(pred) == 1 else "HAM (Real message)"
		confidence = prob[int(pred)] * 100

		print(f"\n  Message {i}: {example}")
		print(f"  >>> Computer says: {label}")
		print(f"  >>> Confidence: {confidence:.1f}%")
		print(f"  >>> Explanation: This looks like a {label.lower()} message")

	print_header("DONE!")
	print_simple_message("The computer has finished learning and testing!")
	print_simple_message("Model saved and ready to use.")