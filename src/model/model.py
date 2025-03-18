import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

def load_data(file_path):
    """Charge les données à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

class NaiveBayesModel:
    def __init__(self, vectorizer=None, use_svd=False, n_components=50):
        self.model = MultinomialNB()
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components) if use_svd else None

    def train(self, df_train):
        """Entraîne le modèle Naive Bayes."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        if self.use_svd:
            X_train_tfidf = self.svd.fit_transform(X_train_tfidf)
        self.model.fit(X_train_tfidf, df_train['label'])

    def evaluate(self, df_test):
        """Évalue le modèle Naive Bayes."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        if self.use_svd:
            X_test_tfidf = self.svd.transform(X_test_tfidf)
        y_pred = self.model.predict(X_test_tfidf)
        return classification_report(df_test['label'], y_pred)

class SVMModel:
    def __init__(self, vectorizer=None, n_components=50):
        self.model = SVC(probability=True)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.param_grid = {
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }

    def train(self, df_train):
        """Entraîne le modèle SVM avec validation croisée."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        X_train_reduced = self.svd.fit_transform(X_train_tfidf)
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_reduced, df_train['label'])
        self.best_model = grid_search.best_estimator_
        print("Meilleurs paramètres pour SVM:", grid_search.best_params_)

    def evaluate(self, df_test):
        """Évalue le modèle SVM."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        X_test_reduced = self.svd.transform(X_test_tfidf)
        y_pred = self.best_model.predict(X_test_reduced)
        return classification_report(df_test['label'], y_pred)

class LogisticRegressionModel:
    def __init__(self, vectorizer=None, use_svd=False, n_components=50):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components) if use_svd else None

    def train(self, df_train):
        """Entraîne le modèle de régression logistique."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        if self.use_svd:
            X_train_tfidf = self.svd.fit_transform(X_train_tfidf)
        self.model.fit(X_train_tfidf, df_train['label'])

    def evaluate(self, df_test):
        """Évalue le modèle de régression logistique."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        if self.use_svd:
            X_test_tfidf = self.svd.transform(X_test_tfidf)
        y_pred = self.model.predict(X_test_tfidf)
        return classification_report(df_test['label'], y_pred)
