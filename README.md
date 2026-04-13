# 📱 SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using TF-IDF vectorization and Logistic Regression.

---

## 📌 Overview

This project trains a binary text classifier on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). The dataset is automatically downloaded if not present locally.

The trained model and vectorizer are saved as `.joblib` files for later reuse.

---

## 🗂️ Project Structure

```
SMSspamclassifier/
├── main.py               # Main training and evaluation script
├── mail_data.csv         # Dataset (auto-downloaded if missing)
├── model.joblib          # Saved trained model (generated after running)
├── vectorizer.joblib     # Saved TF-IDF vectorizer (generated after running)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

1. **Data Loading** — Loads `mail_data.csv` or auto-downloads the UCI SMS Spam Collection
2. **Preprocessing** — Lowercases text, removes URLs and special characters
3. **Vectorization** — TF-IDF with unigrams + bigrams, English stopwords removed
4. **Training** — Logistic Regression classifier
5. **Evaluation** — Prints accuracy, classification report, and confusion matrix
6. **Saving** — Exports `model.joblib` and `vectorizer.joblib` for reuse

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/satorufreaking/SMSspamclassifier.git
cd SMSspamclassifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the classifier

```bash
python main.py
```

If `mail_data.csv` is not present, the script will automatically download the dataset from UCI.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
requests
joblib
```

Install via:

```bash
pip install pandas numpy scikit-learn requests joblib
```

---

## 📊 Model Performance

The model is evaluated on a 20% held-out test set (stratified split).

| Metric    | Value (approx.) |
|-----------|-----------------|
| Accuracy  | ~97–98%         |
| Precision | High for spam   |
| Recall    | High for spam   |

> Exact metrics will be printed in the console when you run the script.

---

## 🧪 Sample Predictions

After training, the script runs two example predictions:

```
- [spam] Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
- [ham]  Hey, are we still meeting for lunch today?
```

---

## 📁 Dataset

- **Name:** SMS Spam Collection
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size:** 5,574 SMS messages (4,827 ham, 747 spam)
- **Format:** Tab-separated — `label` and `text` columns

---

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn** — TF-IDF, Logistic Regression, metrics
- **pandas** — Data loading and processing
- **joblib** — Model serialization
- **requests** — Auto-downloading dataset

---

## 📄 License

This project is open source. Dataset is provided by UCI Machine Learning Repository for research purposes.
