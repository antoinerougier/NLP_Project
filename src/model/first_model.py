import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def load_data(file_path):
    """Charge les données à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

def preprocess_and_train(df):
    """Prétraite les données et entraîne un modèle Naive Bayes."""
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Vectorisation du texte avec TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Entraîner un classificateur Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Prédire les étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test_tfidf)

    # Évaluer les performances du modèle
    report = classification_report(y_test, y_pred)
    print(report)

    return model, vectorizer
