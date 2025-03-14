import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(file_path):
    """Charge les données à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

class NaiveBayesModel:
    def __init__(self, vectorizer=None):
        self.model = MultinomialNB()
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)

    def train(self, df_train):
        """Entraîne le modèle Naive Bayes."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        self.model.fit(X_train_tfidf, df_train['label'])

    def evaluate(self, df_test):
        """Évalue le modèle Naive Bayes."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        y_pred = self.model.predict(X_test_tfidf)
        return classification_report(df_test['label'], y_pred)

class SVMModel:
    def __init__(self, vectorizer=None):
        self.model = SVC(probability=True)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)
        self.param_grid = {
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }

    def train(self, df_train):
        """Entraîne le modèle SVM avec validation croisée."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        grid_search = GridSearchCV(self.model, self.param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_tfidf, df_train['label'])
        self.best_model = grid_search.best_estimator_
        print("Meilleurs paramètres pour SVM:", grid_search.best_params_)

    def evaluate(self, df_test):
        """Évalue le modèle SVM."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        y_pred = self.best_model.predict(X_test_tfidf)
        return classification_report(df_test['label'], y_pred)