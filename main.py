import os
from joblib import dump
from src.pre_processing.download_data import download_and_extract_data
from src.pre_processing.pre_processing import extract_tar
from src.pre_processing.dataframe_creation import create_dataframe
from src.model.model import load_data, TextClassificationModel
from src.visualisation.viz import analyze_data, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def main():
    # Créer le dossier 'data' s'il n'existe pas
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Dossier '{data_dir}' créé.")

    # Chemins
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar"
    filename = os.path.join(data_dir, "aclImdb_v1.tar")
    extract_to = data_dir
    input_pos_train = os.path.join(data_dir, 'aclImdb', 'train', 'pos')
    input_neg_train = os.path.join(data_dir, 'aclImdb', 'train', 'neg')
    input_pos_test = os.path.join(data_dir, 'aclImdb', 'test', 'pos')
    input_neg_test = os.path.join(data_dir, 'aclImdb', 'test', 'neg')
    output_path_train = os.path.join(data_dir, 'data_intermediaire_train.parquet')
    output_path_test = os.path.join(data_dir, 'data_intermediaire_test.parquet')

    # Télécharger et extraire les données
    #download_and_extract_data(url, filename, extract_to)

    # Extraire le fichier tar (si nécessaire, par exemple si le fichier est déjà extrait)
    #extract_tar(filename, extract_to)

    # Créer les DataFrames
    #create_dataframe(input_pos_train, input_neg_train, input_pos_test, input_neg_test, output_path_train, output_path_test)

    # Analyser les données d'entraînement
    #analyze_data(output_path_train)

    # Charger les données d'entraînement et de test
    df_train = load_data(output_path_train)
    df_test = load_data(output_path_test)

    nb_model_without_svd = TextClassificationModel(model=MultinomialNB(), use_svd=False)
    nb_model_without_svd.train(df_train)
    print("Fin modèle Naive Bayes sans réduction")

    svm_model = TextClassificationModel(
        model=SVC(probability=True),
        use_svd=True,
        use_grid_search=True,
        param_grid={
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }
    )
    svm_model.train(df_train)
    print("Fin modèle SVM")

    lr_model = TextClassificationModel(model=LogisticRegression(max_iter=1000), use_svd=True)
    lr_model.train(df_train)
    print("Fin modèle Logistic Regression")

    # Évaluer les modèles
    report_nb_without_svd = nb_model_without_svd.evaluate(df_test)
    report_svm = svm_model.evaluate(df_test)
    report_lr = lr_model.evaluate(df_test)


    #sauvegardons les modeles
    data_dir = "modeles"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Dossier '{data_dir}' créé.")

    dump(nb_model_without_svd, os.path.join(data_dir, 'nb_model.joblib'))
    dump(svm_model, os.path.join(data_dir, 'svm_model.joblib'))
    dump(lr_model, os.path.join(data_dir, 'lr_model.joblib'))

    # Évaluer les modèles et générer les matrices de confusion
    y_test = df_test['label']
    y_pred_nb = nb_model_without_svd.best_model.predict(nb_model_without_svd.vectorizer.transform(df_test['text']))
    y_pred_svm = svm_model.best_model.predict(svm_model.vectorizer.transform(df_test['text']))
    y_pred_lr = lr_model.best_model.predict(lr_model.vectorizer.transform(df_test['text']))

    # Générer et sauvegarder les matrices de confusion
    plot_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes", os.path.join(data_dir, 'confusion_matrix_nb.png'))
    plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - SVM", os.path.join(data_dir, 'confusion_matrix_svm.png'))
    plot_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression", os.path.join(data_dir, 'confusion_matrix_lr.png'))

    # Afficher les rapports de classification
    print("Rapport de classification pour Naive Bayes sans réduction :")
    print(report_nb_without_svd)

    print("Rapport de classification pour SVM :")
    print(report_svm)

    print("Rapport de classification pour Logistic Regression :")
    print(report_lr)

if __name__ == "__main__":
    main()
