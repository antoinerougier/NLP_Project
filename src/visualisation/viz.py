import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect, DetectorFactory
import nltk
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

nltk.download('punkt')

DetectorFactory.seed = 0
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def analyze_data(file_path):

    df = pd.read_parquet(file_path)

    df['language'] = df['text'].apply(detect_language)
    language_counts = df['language'].value_counts()
    print("Distribution des langues détectées :")
    print(language_counts)

    def count_words(text):
        if isinstance(text, str):
            words = text.split()
            return len(words)
        return 0

    df['word_count'] = df['text'].apply(count_words)

    word_count_distribution = df['word_count'].describe()
    print("\nStatistiques sur le nombre de mots :")
    print(word_count_distribution)

    plt.figure(figsize=(10, 6))
    plt.hist(df['word_count'], bins=50, alpha=0.7, color='blue')
    plt.title('Distribution du nombre de mots par revue')
    plt.xlabel('Nombre de mots')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()

    additional_stopwords = set(["the", "a", "an", "and", "is", "it", "of", "to", "in", "this", "that", "with"])
    stopwords = STOPWORDS.union(additional_stopwords)

    for label, category in [(0, "Mauvais film"), (1, "Bon film")]:
        category_data = df[df['label'] == label]['text'].str.cat(sep=' ')
        wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=400).generate(category_data)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {category}")
        plt.show()

    return df


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """Trace et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.show()

def plot_roc_curve_comparison(y_test, y_proba_dict, model_names, output_path):
    """Trace les courbes ROC AUC pour plusieurs modèles sur un seul graphique interactif avec Plotly."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    for model_name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (area = {roc_auc:.2f})'
            )
        )

    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title="Models",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.write_image(output_path)
    fig.show()

def calculate_gini(y_true, y_proba):
    """Calcule le score Gini à partir de l'AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    return 2 * roc_auc - 1
