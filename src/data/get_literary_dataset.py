import os
import requests
import re
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_gutenberg_book(url, output_dir='data/text/gutenberg'):
    """Télécharge un livre du Project Gutenberg à partir d'une URL"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraire l'ID du livre à partir de l'URL
    book_id = url.split('/')[-1].split('.')[0]
    if not book_id.isdigit():
        book_id = re.search(r'/([0-9]+)', url).group(1)
    
    output_file = os.path.join(output_dir, f"{book_id}.txt")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_file):
        logger.info(f"Le livre {book_id} existe déjà.")
        with open(output_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Télécharger le contenu
    logger.info(f"Téléchargement du livre {book_id} depuis {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        content = response.text
        
        # Extraire le texte du livre (entre les balises de début et fin du Project Gutenberg)
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        
        start_pos = content.find(start_marker)
        if start_pos == -1:
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
            start_pos = content.find(start_marker)
        
        end_pos = content.find(end_marker)
        
        if start_pos != -1 and end_pos != -1:
            start_pos = content.find('\n', start_pos) + 1
            text = content[start_pos:end_pos].strip()
            
            # Nettoyer le texte
            text = re.sub(r'\r\n', '\n', text)  # Normaliser les sauts de ligne
            text = re.sub(r'\n{3,}', '\n\n', text)  # Réduire les sauts de ligne multiples
            
            # Enregistrer le fichier nettoyé
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Livre sauvegardé dans {output_file}")
            return text
        else:
            logger.error(f"Impossible de trouver les marqueurs de début/fin dans {url}")
            return None
    else:
        logger.error(f"Échec du téléchargement: {response.status_code}")
        return None

def create_combined_dataset(output_file='data/text/literary_corpus.txt'):
    """Télécharge et combine plusieurs livres classiques pour créer un dataset d'entraînement"""
    # Liste de livres classiques (URL du Project Gutenberg)
    books = [
        'https://www.gutenberg.org/files/1342/1342-0.txt',  # Pride and Prejudice
        'https://www.gutenberg.org/files/84/84-0.txt',      # Frankenstein
        'https://www.gutenberg.org/files/2701/2701-0.txt',  # Moby Dick
        'https://www.gutenberg.org/files/1400/1400-0.txt',  # Great Expectations
        'https://www.gutenberg.org/files/11/11-0.txt',      # Alice in Wonderland
        'https://www.gutenberg.org/files/98/98-0.txt',      # A Tale of Two Cities
        'https://www.gutenberg.org/files/1952/1952-0.txt',  # The Yellow Wallpaper
        'https://www.gutenberg.org/files/1661/1661-0.txt',  # The Adventures of Sherlock Holmes
    ]
    
    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combined_text = ""
    
    # Télécharger et traiter chaque livre
    for url in tqdm(books, desc="Téléchargement des livres"):
        book_text = download_gutenberg_book(url)
        if book_text:
            combined_text += book_text + "\n\n"
    
    # Enregistrer le dataset combiné
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    logger.info(f"Dataset combiné créé avec succès: {output_file} ({len(combined_text)} caractères)")
    return output_file

if __name__ == "__main__":
    create_combined_dataset()