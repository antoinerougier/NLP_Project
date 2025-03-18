import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect, DetectorFactory
import nltk
import seaborn as sns
from sklearn.metrics import confusion_matrix

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