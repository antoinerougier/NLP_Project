import os
import urllib.request
import tarfile

def download_and_extract_data(url, filename, extract_to):
    """Télécharge et extrait les données depuis une URL donnée."""
    # Télécharger le fichier
    print("Téléchargement du fichier...")
    urllib.request.urlretrieve(url, filename)

    # Vérifier si le fichier a été téléchargé
    if not os.path.isfile(filename):
        print(f"Le fichier {filename} n'a pas été téléchargé correctement.")
        return

    # Extraction du fichier tar
    print("Extraction du fichier...")
    with tarfile.open(filename, 'r') as tar:
        tar.extractall(path=extract_to)
        print(f"Extraction terminée dans le répertoire {extract_to}")
