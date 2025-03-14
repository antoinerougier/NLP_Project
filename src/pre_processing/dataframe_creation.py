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

def create_dataframe(input_pos_train, input_neg_train, input_pos_test, input_neg_test, output_path_train, output_path_test):
    """Crée des DataFrames à partir des revues positives et négatives et les sauvegarde en format Parquet."""
    pos_reviews_train = load_reviews(input_pos_train, 'pos')
    neg_reviews_train = load_reviews(input_neg_train, 'neg')
    pos_reviews_test = load_reviews(input_pos_test, 'pos')
    neg_reviews_test = load_reviews(input_neg_test, 'neg')

    df_train = pd.DataFrame(pos_reviews_train + neg_reviews_train)
    df_test = pd.DataFrame(pos_reviews_test + neg_reviews_test)

    df_train['label'] = df_train['label'].map({'pos': 1, 'neg': 0})
    df_test['label'] = df_test['label'].map({'pos': 1, 'neg': 0})

    df_train.to_parquet(output_path_train, index=False)
    df_test.to_parquet(output_path_test, index=False)

    print(f"DataFrame d'entraînement sauvegardé en tant que {output_path_train}")
    print(f"DataFrame de test sauvegardé en tant que {output_path_test}")
