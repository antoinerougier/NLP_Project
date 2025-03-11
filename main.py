import os

from src.pre_processing.download_data import download_and_extract_data
from src.pre_processing.pre_processing import extract_tar
from src.pre_processing.dataframe_creation import create_dataframe
from src.model.first_model import load_data, preprocess_and_train


def main():
    # Chemins
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar"
    filename = "data/aclImdb_v1.tar"
    extract_to = "data"
    input_pos = os.path.join('data', 'aclImdb', 'train', 'pos')
    input_neg = os.path.join('data', 'aclImdb', 'train', 'neg')
    output_path = os.path.join('data', 'data_intermediaire.parquet')

    # Télécharger et extraire les données
    download_and_extract_data(url, filename, extract_to)

    # Extraire le fichier tar (si nécessaire, par exemple si le fichier est déjà extrait)
    extract_tar(filename, extract_to)

    # Créer le DataFrame
    create_dataframe(input_pos, input_neg, output_path)

    # Charger les données
    df = load_data(output_path)

    # Prétraiter les données et entraîner le modèle
    model, vectorizer, report = preprocess_and_train(df)

    # Afficher le rapport de classification
    print("Rapport de classification :")
    print(report)

if __name__ == "__main__":
    main()
