import os
import pandas as pd

def load_reviews(directory, label):
    """Charge les fichiers texte d'un répertoire et les étiquette."""
    reviews = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    reviews.append({'text': text, 'label': label})
    else:
        print(f"Le répertoire {directory} n'existe pas.")
    return reviews

def create_dataframe(input_pos, input_neg, output_path):
    """Crée un DataFrame à partir des revues positives et négatives et le sauvegarde en format Parquet."""
    pos_reviews = load_reviews(input_pos, 'pos')
    neg_reviews = load_reviews(input_neg, 'neg')

    df = pd.DataFrame(pos_reviews + neg_reviews)
    df['label'] = df['label'].map({'pos': 1, 'neg': 0})

    df.to_parquet(output_path, index=False)
    print(f"DataFrame sauvegardé en tant que {output_path}")
