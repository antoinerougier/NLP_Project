import os

from src.pre_processing.download_data import download_and_extract_data
from src.pre_processing.pre_processing import extract_tar
from src.pre_processing.dataframe_creation import create_dataframe
from src.model.model import load_data, NaiveBayesModel, SVMModel
from src.visualisation.viz import analyze_data 

def main():
    # Chemins
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar"
    filename = "data/aclImdb_v1.tar"
    extract_to = "data"
    input_pos_train = os.path.join('data', 'aclImdb', 'train', 'pos')
    input_neg_train = os.path.join('data', 'aclImdb', 'train', 'neg')
    input_pos_test = os.path.join('data', 'aclImdb', 'test', 'pos')
    input_neg_test = os.path.join('data', 'aclImdb', 'test', 'neg')
    output_path_train = os.path.join('data', 'data_intermediaire_train.parquet')
    output_path_test = os.path.join('data', 'data_intermediaire_test.parquet')

    # Télécharger et extraire les données
    download_and_extract_data(url, filename, extract_to)

    # Extraire le fichier tar (si nécessaire, par exemple si le fichier est déjà extrait)
    extract_tar(filename, extract_to)

    # Créer les DataFrames
    create_dataframe(input_pos_train, input_neg_train, input_pos_test, input_neg_test, output_path_train, output_path_test)

    # Analyser les données d'entraînement
    analyze_data(output_path_train)

    # Charger les données d'entraînement et de test
    df_train = load_data(output_path_train)
    df_test = load_data(output_path_test)

    # Initialiser et entraîner les modèles
    nb_model = NaiveBayesModel()
    nb_model.train(df_train)

    svm_model = SVMModel()
    svm_model.train(df_train)

    # Évaluer les modèles
    report_nb = nb_model.evaluate(df_test)
    report_svm = svm_model.evaluate(df_test)

    # Afficher les rapports de classification
    print("Rapport de classification pour Naive Bayes :")
    print(report_nb)

    print("Rapport de classification pour SVM :")
    print(report_svm)

if __name__ == "__main__":
    main()