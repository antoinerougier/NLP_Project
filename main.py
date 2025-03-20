import os
from joblib import dump
from src.pre_processing.download_data import download_and_extract_data
from src.pre_processing.pre_processing import extract_tar
from src.pre_processing.dataframe_creation import create_dataframe
from src.model.model import load_data, TextClassificationModel
from src.visualisation.viz import plot_confusion_matrix, plot_roc_curve, calculate_gini
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
    download_and_extract_data(url, filename, extract_to)

    # Extraire le fichier tar (si nécessaire, par exemple si le fichier est déjà extrait)
    extract_tar(filename, extract_to)

    # Créer les DataFrames
    create_dataframe(input_pos_train, input_neg_train, input_pos_test, input_neg_test, output_path_train, output_path_test)

    # Charger les données d'entraînement et de test
    df_train = load_data(output_path_train)
    df_test = load_data(output_path_test)

    # Définir les paramètres de grille pour Naive Bayes et Logistic Regression
    nb_param_grid = {'alpha': [0.1, 1.0, 10.0, 20.0]}
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale'],
        'degree': [2, 3],  
        'coef0': [0.0, 0.1, 0.5]  
    }
    lr_param_grid = {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs']}

    # Initialiser et entraîner les modèles
    nb_model = TextClassificationModel(model=MultinomialNB(), use_svd=False, use_grid_search=True, param_grid=nb_param_grid)
    nb_model.train(df_train)
    print("Fin modèle Naive Bayes avec GridSearch")

    svm_model = TextClassificationModel(
        model=SVC(probability=True),
        use_svd=True,
        use_grid_search=True,
        param_grid=svm_param_grid
    )
    svm_model.train(df_train)
    print("Fin modèle SVM")

    lr_model = TextClassificationModel(model=LogisticRegression(max_iter=1000), use_svd=True, use_grid_search=True, param_grid=lr_param_grid)
    lr_model.train(df_train)
    print("Fin modèle Logistic Regression avec GridSearch")
    # Évaluer les modèles
    report_nb = nb_model.evaluate(df_test)
    report_svm = svm_model.evaluate(df_test)
    report_lr = lr_model.evaluate(df_test)

    # Sauvegarder les modèles
    model_dir = "modeles"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Dossier '{model_dir}' créé.")

    dump(nb_model, os.path.join(model_dir, 'nb_model.joblib'))
    dump(svm_model, os.path.join(model_dir, 'svm_model.joblib'))
    dump(lr_model, os.path.join(model_dir, 'lr_model.joblib'))

    # Évaluer les modèles et générer les matrices de confusion
    y_test = df_test['label']
    y_pred_nb = nb_model.best_model.predict(nb_model.vectorizer.transform(df_test['text']))
    y_pred_svm = svm_model.best_model.predict(svm_model.svd.transform(svm_model.vectorizer.transform(df_test['text'])))
    y_pred_lr = lr_model.best_model.predict(lr_model.svd.transform(lr_model.vectorizer.transform(df_test['text'])))

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Dossier '{output_dir}' créé.")

    # Générer et sauvegarder les matrices de confusion
    plot_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes", os.path.join(output_dir, 'confusion_matrix_nb.png'))
    plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - SVM", os.path.join(output_dir, 'confusion_matrix_svm.png'))
    plot_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression", os.path.join(output_dir, 'confusion_matrix_lr.png'))

    # Générer et sauvegarder les courbes ROC AUC
    y_proba_nb = nb_model.best_model.predict_proba(nb_model.vectorizer.transform(df_test['text']))[:, 1]
    y_proba_svm = svm_model.best_model.predict_proba(svm_model.svd.transform(svm_model.vectorizer.transform(df_test['text'])))[:, 1]
    y_proba_lr = lr_model.best_model.predict_proba(lr_model.svd.transform(lr_model.vectorizer.transform(df_test['text'])))[:, 1]

    roc_auc_nb = plot_roc_curve(y_test, y_proba_nb, "ROC Curve - Naive Bayes", os.path.join(output_dir, 'roc_curve_nb.png'))
    roc_auc_svm = plot_roc_curve(y_test, y_proba_svm, "ROC Curve - SVM", os.path.join(output_dir, 'roc_curve_svm.png'))
    roc_auc_lr = plot_roc_curve(y_test, y_proba_lr, "ROC Curve - Logistic Regression", os.path.join(output_dir, 'roc_curve_lr.png'))

    # Calculer et afficher les scores Gini
    gini_nb = calculate_gini(roc_auc_nb)
    gini_svm = calculate_gini(roc_auc_svm)
    gini_lr = calculate_gini(roc_auc_lr)

    print(f"Score Gini pour Naive Bayes: {gini_nb:.2f}")
    print(f"Score Gini pour SVM: {gini_svm:.2f}")
    print(f"Score Gini pour Logistic Regression: {gini_lr:.2f}")

    # Afficher les rapports de classification
    print("Rapport de classification pour Naive Bayes :")
    print(report_nb)

    print("Rapport de classification pour SVM :")
    print(report_svm)

    print("Rapport de classification pour Logistic Regression :")
    print(report_lr)

if __name__ == "__main__":
    main()