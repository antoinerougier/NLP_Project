from src.model.first_model import load_data, preprocess_and_train

def main():
    # Chemin vers le fichier Parquet
    file_path = 'data/data_intermediaire.parquet'

    # Charger les données
    df = load_data(file_path)

    # Prétraiter les données et entraîner le modèle
    model, vectorizer = preprocess_and_train(df)

if __name__ == "__main__":
    main()