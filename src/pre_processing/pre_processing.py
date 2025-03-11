import tarfile
import os

def extract_tar(tar_path, extract_to):
    """Extrait le contenu d'un fichier tar dans un répertoire spécifié."""
    if not os.path.isfile(tar_path):
        print(f"Le fichier {tar_path} n'existe pas.")
        return

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
        print(f"Extraction terminée dans le répertoire {extract_to}")
