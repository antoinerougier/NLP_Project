import tarfile
import os

def extract_tar(tar_path, extract_to):
    # Vérifiez si le fichier tar existe
    if not os.path.isfile(tar_path):
        print(f"Le fichier {tar_path} n'existe pas.")
        return

    # Ouvrez le fichier tar
    with tarfile.open(tar_path, 'r') as tar:
        # Extrait le contenu dans le répertoire spécifié
        tar.extractall(path=extract_to)
        print(f"Extraction terminée dans le répertoire {extract_to}")

if __name__ == "__main__":
    # Chemin vers le fichier tar
    tar_path = os.path.join('data/aclImdb_v1.tar')
    # Répertoire où extraire le contenu
    extract_to = os.path.join('data')

    # Impression des chemins pour vérification
    print(f"Chemin du fichier tar : {tar_path}")
    print(f"Répertoire d'extraction : {extract_to}")

    extract_tar(tar_path, extract_to)
