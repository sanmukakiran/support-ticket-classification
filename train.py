import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 1. Load Data
print("\nLoading dataset...")
df = pd.read_csv("customer_support_tickets.csv")

# We only need Ticket Description, Ticket Type, and Ticket Priority
df = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']].dropna()

print(f"Dataset loaded. Total shape: {df.shape}")

# 2. Text Preprocessing
print("Initializing preprocessing tools...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize (simple split)
    tokens = text.split()
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

print("Applying text cleaning (this might take a minute)...")
df['Clean_Text'] = df['Ticket Description'].apply(clean_text)

# 3. Feature Extraction (TF-IDF)
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Clean_Text'])

y_category = df['Ticket Type']
y_priority = df['Ticket Priority']

print("Splitting data into training and test sets...")
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42)
X_train_prio, X_test_prio, y_train_prio, y_test_prio = train_test_split(X, y_priority, test_size=0.2, random_state=42)

# 4. Model Training
print("Training Ticket Category Model (Logistic Regression)...")
model_cat = LogisticRegression(max_iter=1000, random_state=42)
model_cat.fit(X_train_cat, y_train_cat)

print("Training Ticket Priority Model (Random Forest)...")
model_prio = RandomForestClassifier(n_estimators=100, random_state=42) # Priority usually benefits from robust trees
model_prio.fit(X_train_prio, y_train_prio)

# 5. Evaluation & Reports
print("\n--- CATEGORY CLASSIFICATION REPORT ---")
y_pred_cat = model_cat.predict(X_test_cat)
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print(classification_report(y_test_cat, y_pred_cat))

print("\n--- PRIORITY CLASSIFICATION REPORT ---")
y_pred_prio = model_prio.predict(X_test_prio)
print("Accuracy:", accuracy_score(y_test_prio, y_pred_prio))
print(classification_report(y_test_prio, y_pred_prio))

print("\nSaving completed models and vectorizer...")
joblib.dump(model_cat, 'category_model.pkl')
joblib.dump(model_prio, 'priority_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Pipeline execution complete!")
