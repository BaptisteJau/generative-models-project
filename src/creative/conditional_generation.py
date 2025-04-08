import os
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from src.utils.visualization import plot_generated_images
from src.creative.latent_exploration import LatentExplorer

logger = logging.getLogger(__name__)

class AttributeController:
    """
    Contrôleur d'attributs pour la génération conditionnelle
    
    Cette classe découvre et manipule des attributs latents dans le modèle GAN
    pour permettre la génération conditionnelle d'images avec des caractéristiques
    spécifiques.
    """
    
    def __init__(self, gan_model, latent_explorer=None, output_dir="results/conditional_generation"):
        """
        Initialise le contrôleur d'attributs
        
        Args:
            gan_model: Instance d'un modèle GAN (typiquement DeepCNN)
            latent_explorer: Instance de LatentExplorer (optionnel, sera créé si None)
            output_dir: Répertoire pour sauvegarder les résultats
        """
        self.model = gan_model
        self.latent_explorer = latent_explorer or LatentExplorer(gan_model, output_dir=output_dir)
        self.output_dir = output_dir
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Stockage des directions d'attributs découvertes
        self.attribute_directions = {}
    
    def find_attribute_direction(self, positive_samples, negative_samples=None, name=None, save_path=None):
        """
        Trouve une direction dans l'espace latent correspondant à un attribut
        
        Args:
            positive_samples: Liste d'indices ou vecteurs latents avec l'attribut
            negative_samples: Liste d'indices ou vecteurs latents sans l'attribut
            name: Nom de l'attribut (optionnel)
            save_path: Chemin pour sauvegarder la visualisation
            
        Returns:
            Vecteur de direction de l'attribut
        """
        # Si les échantillons sont des indices, les convertir en vecteurs latents
        pos_vectors = self._convert_to_vectors(positive_samples)
        
        if negative_samples is not None:
            neg_vectors = self._convert_to_vectors(negative_samples)
            
            # La direction est la différence moyenne entre les groupes positif et négatif
            pos_mean = np.mean(pos_vectors, axis=0)
            neg_mean = np.mean(neg_vectors, axis=0)
            direction = pos_mean - neg_mean
            
        else:
            # Direction par rapport à la moyenne des vecteurs
            pos_mean = np.mean(pos_vectors, axis=0)
            
            # Générer une base de référence (moyenne de vecteurs aléatoires)
            n_random = len(pos_vectors)
            random_vectors = []
            
            for _ in range(n_random):
                random_vectors.append(self.latent_explorer.random_latent_vector().numpy().flatten())
                
            baseline_mean = np.mean(random_vectors, axis=0)
            direction = pos_mean - baseline_mean
            
        # Normaliser la direction
        direction = direction / np.linalg.norm(direction)
        
        # Sauvegarder la direction si un nom est fourni
        if name:
            self.attribute_directions[name] = direction
            
        # Visualiser l'effet de cette direction
        if save_path or name:
            # Si uniquement le nom est fourni, créer un chemin de sauvegarde
            if not save_path and name:
                save_path = os.path.join(self.output_dir, f"attribute_{name}.png")
            
            # Créer un vecteur de base aléatoire
            base_vector = self.latent_explorer.random_latent_vector().numpy().flatten()
            
            # Générer des images avec différentes intensités de l'attribut
            values = np.linspace(-3, 3, 7)
            images = []
            
            for value in values:
                vector = base_vector + direction * value
                img = self.latent_explorer.generate_from_vector(vector)
                images.append(img[0])
                
            # Visualiser
            fig, axes = plt.subplots(1, len(images), figsize=(14, 3))
            title = f"Attribut: {name}" if name else "Direction d'attribut découverte"
            plt.suptitle(title, fontsize=14)
            
            for i, (img, val) in enumerate(zip(images, values)):
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    axes[i].imshow(img.squeeze(), cmap='gray')
                else:
                    axes[i].imshow(img)
                    
                axes[i].set_title(f"Valeur: {val:.1f}")
                axes[i].axis('off')
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Visualisation sauvegardée: {save_path}")
                
            plt.show()
            
        return direction
            
    def _convert_to_vectors(self, samples):
        """
        Convertit des échantillons en vecteurs latents
        
        Args:
            samples: Liste d'indices ou de vecteurs latents
            
        Returns:
            Liste de vecteurs latents numpy
        """
        vectors = []
        
        for sample in samples:
            if isinstance(sample, int):
                # C'est un indice/seed, générer un vecteur aléatoire
                np.random.seed(sample)
                vector = np.random.normal(0, 1, self.latent_explorer.latent_dim)
            elif isinstance(sample, (list, np.ndarray, tf.Tensor, torch.Tensor)):
                # C'est déjà un vecteur
                if isinstance(sample, (tf.Tensor, torch.Tensor)):
                    vector = sample.numpy() if isinstance(sample, tf.Tensor) else sample.cpu().numpy()
                else:
                    vector = np.array(sample)
            else:
                raise ValueError(f"Type d'échantillon non reconnu: {type(sample)}")
                
            # S'assurer que le vecteur est aplati
            vectors.append(vector.flatten())
            
        return vectors
    
    def apply_attribute(self, base_vector, attribute, intensity=1.0):
        """
        Applique un attribut à un vecteur latent
        
        Args:
            base_vector: Vecteur latent de base
            attribute: Nom d'attribut (str) ou direction d'attribut (array)
            intensity: Intensité de l'attribut à appliquer
            
        Returns:
            Vecteur latent modifié
        """
        # Convertir base_vector en numpy array si nécessaire
        if isinstance(base_vector, (tf.Tensor, torch.Tensor)):
            base_vector = base_vector.numpy() if isinstance(base_vector, tf.Tensor) else base_vector.cpu().numpy()
        
        # Récupérer la direction d'attribut
        if isinstance(attribute, str):
            if attribute not in self.attribute_directions:
                raise ValueError(f"Attribut '{attribute}' non trouvé. "
                               f"Disponibles: {list(self.attribute_directions.keys())}")
            direction = self.attribute_directions[attribute]
        else:
            direction = attribute
            
        # Appliquer la direction avec l'intensité spécifiée
        modified_vector = base_vector + direction * intensity
        
        return modified_vector
    
    def generate_with_attributes(self, attributes_dict, base_vector=None, save_path=None):
        """
        Génère une image avec des attributs spécifiques
        
        Args:
            attributes_dict: Dictionnaire {nom_attribut: intensité}
            base_vector: Vecteur latent de base (None pour aléatoire)
            save_path: Chemin pour sauvegarder l'image générée
            
        Returns:
            Image générée et vecteur latent utilisé
        """
        # Créer ou utiliser un vecteur de base
        if base_vector is None:
            base_vector = self.latent_explorer.random_latent_vector().numpy().flatten()
            
        # Appliquer chaque attribut
        modified_vector = base_vector.copy()
        
        for attr, intensity in attributes_dict.items():
            modified_vector = self.apply_attribute(modified_vector, attr, intensity)
            
        # Générer l'image
        generated_img = self.latent_explorer.generate_from_vector(modified_vector)
        
        # Visualiser et sauvegarder si demandé
        if save_path:
            img = generated_img[0]
            
            plt.figure(figsize=(5, 5))
            
            if img.shape[0] == 1 or img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
                
            if img.shape[-1] == 1:
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
                
            # Créer un titre avec les attributs appliqués
            attr_str = ", ".join([f"{k}: {v:.2f}" for k, v in attributes_dict.items()])
            plt.title(f"Image générée avec attributs:\n{attr_str}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            logger.info(f"Image sauvegardée: {save_path}")
            
        return generated_img, modified_vector
    
    def interactive_exploration(self, attributes=None, n_attributes=3, save_dir=None):
        """
        Crée une grille d'exploration interactive pour manipuler plusieurs attributs
        
        Args:
            attributes: Liste des noms d'attributs à explorer (None pour utiliser découverts)
            n_attributes: Nombre d'attributs à découvrir si attributes=None
            save_dir: Répertoire pour sauvegarder les résultats
            
        Returns:
            Dictionnaire de résultats de l'exploration
        """
        # Si aucun attribut spécifié, découvrir les principaux attributs
        if attributes is None:
            if len(self.attribute_directions) > 0:
                # Utiliser les attributs déjà découverts
                attributes = list(self.attribute_directions.keys())[:n_attributes]
            else:
                # Découvrir de nouvelles directions via PCA
                logger.info(f"Découverte automatique de {n_attributes} attributs via PCA...")
                results = self.latent_explorer.explore_principal_directions(
                    n_components=n_attributes, 
                    save_dir=os.path.join(self.output_dir, "pca_attributes") if save_dir else None
                )
                
                # Utiliser les directions principales comme attributs
                for i, result in results.items():
                    attr_name = f"attribute_{i+1}"
                    self.attribute_directions[attr_name] = result['direction']
                    
                attributes = [f"attribute_{i+1}" for i in range(n_attributes)]
                
        # Créer un vecteur de base aléatoire
        base_vector = self.latent_explorer.random_latent_vector().numpy().flatten()
        
        # Créer des variations pour chaque attribut
        intensities = np.linspace(-2, 2, 5)
        results = {}
        
        for attr in attributes:
            attr_images = []
            
            for intensity in intensities:
                vector = self.apply_attribute(base_vector, attr, intensity)
                img = self.latent_explorer.generate_from_vector(vector)
                attr_images.append(img[0])
                
            results[attr] = {
                'images': attr_images,
                'intensities': intensities
            }
            
        # Visualiser les résultats
        fig, axes = plt.subplots(len(attributes), len(intensities), 
                                figsize=(len(intensities) * 2, len(attributes) * 2))
        
        for i, attr in enumerate(attributes):
            for j, intensity in enumerate(intensities):
                img = results[attr]['images'][j]
                
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    axes[i, j].imshow(img.squeeze(), cmap='gray')
                else:
                    axes[i, j].imshow(img)
                    
                if i == 0:
                    axes[i, j].set_title(f"{intensity:.1f}")
                    
                if j == 0:
                    axes[i, j].set_ylabel(attr, rotation=45, ha='right')
                    
                axes[i, j].axis('off')
                
        plt.suptitle("Exploration des attributs", fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "attributes_exploration.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Exploration sauvegardée: {save_path}")
            
        plt.show()
        
        return results