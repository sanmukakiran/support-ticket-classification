import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown and code cells
cells = [
    nbf.v4.new_markdown_cell("# 🎫 Support Ticket Classification & Prioritization\nThis notebook demonstrates an end-to-end Machine Learning pipeline for automatically classifying customer support tickets and assigning them a priority level."),
    
    nbf.v4.new_markdown_cell("## 1. Setup and Imports\nWe use standard NLP and ML libraries: `pandas`, `nltk`, and `scikit-learn`."),
    nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport re\nimport nltk\nfrom nltk.corpus import stopwords\nfrom nltk.stem import WordNetLemmatizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Download resources for lemmatization and stopwords\nnltk.download('stopwords', quiet=True)\nnltk.download('wordnet', quiet=True)\nnltk.download('omw-1.4', quiet=True)"""),
    
    nbf.v4.new_markdown_cell("## 2. Loading the Dataset\nWe use the Kaggle 'Customer Support Ticket Dataset'. It contains ticket subject, description, assigned category, and priority."),
    nbf.v4.new_code_cell("df = pd.read_csv('customer_support_tickets.csv')\n# Keep relevant columns\ndf = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']].dropna()\nprint(f'Total tickets loaded: {df.shape[0]}')\ndf.head()"),
    
    nbf.v4.new_markdown_cell("## 3. Data Preprocessing & NLP\nWe clean the text data by lowercasing, removing punctuation, stopwords, and applying lemmatization. This turns raw user input into normalized text."),
    nbf.v4.new_code_cell("""stop_words = set(stopwords.words('english'))\nlemmatizer = WordNetLemmatizer()\n\ndef clean_text(text):\n    if not isinstance(text, str): return ''\n    text = text.lower()\n    text = re.sub(r'[^a-zA-Z\s]', '', text)\n    tokens = text.split()\n    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]\n    return ' '.join(cleaned)\n\ndf['Clean_Text'] = df['Ticket Description'].apply(clean_text)\ndf[['Ticket Description', 'Clean_Text']].head()"""),
    
    nbf.v4.new_markdown_cell("## 4. Feature Extraction & Train/Test Split\nMachine learning models require mathematical representations of text. We use `TfidfVectorizer` to convert cleaned text into TF-IDF scores. Then we split our data into 80% training and 20% testing."),
    nbf.v4.new_code_cell("""tfidf = TfidfVectorizer(max_features=5000)\nX = tfidf.fit_transform(df['Clean_Text'])\n\ny_cat = df['Ticket Type']\ny_prio = df['Ticket Priority']\n\n# Split for Category model\nX_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)\n\n# Split for Priority model\nX_train_prio, X_test_prio, y_train_prio, y_test_prio = train_test_split(X, y_prio, test_size=0.2, random_state=42)"""),

    nbf.v4.new_markdown_cell("## 5. Model Training\nWe train two separate models:\n1. **Categorization**: Logistic Regression (works well for text classification).\n2. **Prioritization**: Random Forest Classifier (handles non-linear correlations better, good for priority assignments)."),
    nbf.v4.new_code_cell("""# 1. Train Category Model\ncat_model = LogisticRegression(max_iter=1000, random_state=42)\ncat_model.fit(X_train_cat, y_train_cat)\n\n# 2. Train Priority Model\nprio_model = RandomForestClassifier(n_estimators=100, random_state=42)\nprio_model.fit(X_train_prio, y_train_prio)"""),
    
    nbf.v4.new_markdown_cell("## 6. Evaluation\nLet's evaluate the models using accuracy, precision, and recall."),
    nbf.v4.new_code_cell("""print('--- CATEGORY CLASSIFICATION REPORT ---')\ny_pred_cat = cat_model.predict(X_test_cat)\nprint(f'Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.2f}\\n')\nprint(classification_report(y_test_cat, y_pred_cat))\n\nprint('\\n--- PRIORITY CLASSIFICATION REPORT ---')\ny_pred_prio = prio_model.predict(X_test_prio)\nprint(f'Accuracy: {accuracy_score(y_test_prio, y_pred_prio):.2f}\\n')\nprint(classification_report(y_test_prio, y_pred_prio))"""),
    
    nbf.v4.new_markdown_cell("### Confusion Matrices\nA confusion matrix shows where the model is confused (predicting one class when it is actually another)."),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n\nsns.heatmap(confusion_matrix(y_test_cat, y_pred_cat), annot=True, fmt='d', cmap='Blues', ax=axes[0],\n            xticklabels=cat_model.classes_, yticklabels=cat_model.classes_)\naxes[0].set_title('Category Confusion Matrix')\naxes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')\n\nsns.heatmap(confusion_matrix(y_test_prio, y_pred_prio), annot=True, fmt='d', cmap='Reds', ax=axes[1],\n            xticklabels=prio_model.classes_, yticklabels=prio_model.classes_)\naxes[1].set_title('Priority Confusion Matrix')\naxes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')\n\nplt.tight_layout()\nplt.show()"""),

    nbf.v4.new_markdown_cell("## 7. Real-time Inference Example\nA function to test how a new support ticket would be handled in production:"),
    nbf.v4.new_code_cell("""def predict_ticket(text):\n    cleaned = clean_text(text)\n    vec = tfidf.transform([cleaned])\n    category = cat_model.predict(vec)[0]\n    priority = prio_model.predict(vec)[0]\n    return {'Category': category, 'Priority': priority}\n\n# Test Examples\nprint("Ticket 1:", predict_ticket("I want to cancel my subscription immediately!"))\nprint("Ticket 2:", predict_ticket("The screen on my newly purchased device is cracked, how do I return it?"))\nprint("Ticket 3:", predict_ticket("Where can I find my invoice for last month?"))""")
]

nb['cells'] = cells

with open('ticket_classification.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
