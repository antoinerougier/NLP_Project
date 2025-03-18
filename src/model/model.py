import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Charge les données à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

class TextClassificationModel:
    def __init__(self, model, vectorizer=None, use_svd=False, use_grid_search=False, n_components=50, param_grid=None):
        self.model = model
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer(stop_words='english', max_features=5000)
        self.use_svd = use_svd
        self.use_grid_search = use_grid_search
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components) if use_svd else None
        self.normalizer = Normalizer() if use_svd else None
        self.param_grid = param_grid
        self.best_model = None

    def train(self, df_train):
        """Entraîne le modèle de classification de texte."""
        X_train_tfidf = self.vectorizer.fit_transform(df_train['text'])
        if self.use_svd:
            X_train_tfidf = self.svd.fit_transform(X_train_tfidf)
            X_train_tfidf = self.normalizer.fit_transform(X_train_tfidf)

        if self.use_grid_search and self.param_grid:
            grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_tfidf, df_train['label'])
            self.best_model = grid_search.best_estimator_
            print("Meilleurs paramètres:", grid_search.best_params_)
        else:
            self.best_model = self.model
            self.best_model.fit(X_train_tfidf, df_train['label'])

    def evaluate(self, df_test):
        """Évalue le modèle de classification de texte."""
        X_test_tfidf = self.vectorizer.transform(df_test['text'])
        if self.use_svd:
            X_test_tfidf = self.svd.transform(X_test_tfidf)
            X_test_tfidf = self.normalizer.transform(X_test_tfidf)
        y_pred = self.best_model.predict(X_test_tfidf)
        return classification_report(df_test['label'], y_pred)