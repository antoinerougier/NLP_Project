# Projet d’Analyse de Sentiments de Films

## Vue d’ensemble

Ce projet porte sur l’analyse de sentiments dans des critiques de films. Nous utilisons des techniques de traitement du langage naturel (NLP) pour analyser et prédire les sentiments exprimés dans ces critiques. Le projet est basé sur les méthodologies présentées dans l’article *"Learning Word Vectors for Sentiment Analysis"* par Andrew L. Maas et al. 

Le rapport se trouve dans le dossier article sous le nom rapport.pdf

## Instructions de Configuration

### Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/antoinerougier/NLP_Project.git
   cd NLP_Project
   ```

   ```bash
   python -m venv nlp
   source nlp/bin/activate
   ```

   ```bash
   pip install -r requirements.txt
   ```


## Données 

Le jeu de données est fourni au format .tar. L’étape de prétraitement consiste à télécharger et extraire ces données.

## Reproductibilité :

Pour exécuter le projet, il suffit de lancer le script main.py. Ce script prend en charge l’ensemble du processus, notamment :

- Le téléchargement du jeu de données
- L’extraction et la mise en forme des données
- L’entraînement du modèle d’analyse de sentiments
- L’évaluation des performances du modèle

   ```bash
   python main.py
   ```

## Structure du projet 

src/preprocessing/download_data.py : Script pour télécharger et extraire le jeu de données.
src/preprocessing/pre_processing.py : Script pour le prétraitement supplémentaire des données.
src/preprocessing/dataframe_creation.py : Script pour créer un DataFrame à partir des données traitées.
src/model/model.py : Script contenant la logique d’entraînement et d’évaluation du modèle.
main.py : Le script principal pour exécuter l’ensemble du pipeline (le code peut être long 2h : causé par le gridsearchCV).

## Notebook 

Le dossier Notebook contient des notebooks Jupyter utilisés pour la visualisation des données et aussi pour tester le Fine-tuning d'un modèle LLM. Ces notebooks ne sont pas essentiels pour exécuter le projet principal.
