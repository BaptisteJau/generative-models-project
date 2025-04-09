import os
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import logging
from tqdm import tqdm

from src.utils.visualization import plot_generated_images, compare_real_generated
from src.utils.framework_utils import FrameworkBridge

logger = logging.getLogger(__name__)

class LatentExplorer:
    """
    Classe permettant d'explorer l'espace latent d'un modèle GAN
    et de créer des variations contrôlées d'images générées
    """
    
    def __init__(self, model, device=None, output_dir="results/creative_exploration"):
        """
        Initialise l'explorateur de l'espace latent
        
        Args:
            model: Instance d'un modèle GAN (typiquement DeepCNN)
            device: Appareil à utiliser ('cuda', 'cpu')
            output_dir: Répertoire pour sauvegarder les résultats
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.latent_dim = getattr(model, 'latent_dim', 100)
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Détecter automatiquement le framework (TensorFlow ou PyTorch)
        self.is_tensorflow = hasattr(model, 'generator') and isinstance(model.generator, tf.keras.Model)
        logger.info(f"Modèle détecté: {'TensorFlow' if self.is_tensorflow else 'PyTorch'} "
                   f"avec dimension latente {self.latent_dim}")
    
    def generate_from_vector(self, latent_vector, as_tensor=False):
        """
        Génère une image à partir d'un vecteur latent spécifique
        
        Args:
            latent_vector: Vecteur latent (numpy array ou tensor)
            as_tensor: Si True, renvoie un tensor, sinon numpy array
            
        Returns:
            Image générée
        """
        # S'assurer que le vecteur a la bonne forme
        if isinstance(latent_vector, list):
            latent_vector = np.array(latent_vector)
            
        if len(latent_vector.shape) == 1:
            latent_vector = latent_vector.reshape(1, -1)
            
        # Générer l'image
        if self.is_tensorflow:
            # Pour TensorFlow (DeepCNN)
            if not isinstance(latent_vector, tf.Tensor):
                latent_vector = tf.convert_to_tensor(latent_vector, dtype=tf.float32)
            
            generated_img = self.model.generator(latent_vector, training=False)
            
            # Normaliser l'image si nécessaire
            if tf.reduce_min(generated_img) < 0:
                generated_img = (generated_img + 1) / 2
                
            if not as_tensor:
                generated_img = generated_img.numpy()
                
            return generated_img
            
        else:
            # Pour PyTorch
            if not isinstance(latent_vector, torch.Tensor):
                latent_vector = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)
                
            with torch.no_grad():
                generated_img = self.model.generator(latent_vector)
                
                # Normaliser l'image si nécessaire
                if generated_img.min() < 0:
                    generated_img = (generated_img + 1) / 2
                
                if not as_tensor:
                    generated_img = generated_img.cpu().numpy()
                    
                return generated_img
    
    def random_latent_vector(self, batch_size=1):
        """
        Génère un vecteur latent aléatoire
        
        Args:
            batch_size: Nombre de vecteurs à générer
            
        Returns:
            Vecteur(s) latent(s) aléatoire(s)
        """
        if self.is_tensorflow:
            return tf.random.normal([batch_size, self.latent_dim])
        else:
            return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def explore_dimension(self, dimension, values=None, n_steps=10, base_vector=None, 
                         other_dims_range=(-1, 1), save_path=None):
        """
        Explore l'effet de la variation d'une dimension spécifique
        
        Args:
            dimension: Index de la dimension à explorer
            values: Liste des valeurs à utiliser (None pour créer automatiquement)
            n_steps: Nombre de pas pour l'exploration (si values=None)
            base_vector: Vecteur de base (None pour générer aléatoirement)
            other_dims_range: Plage de valeurs pour les autres dimensions
            save_path: Chemin pour sauvegarder la visualisation
            
        Returns:
            Liste des images générées
        """
        # Créer ou utiliser le vecteur de base
        if base_vector is None:
            base_vector = np.random.uniform(
                other_dims_range[0], other_dims_range[1], 
                size=self.latent_dim
            )
        
        # Créer la séquence de valeurs pour la dimension à explorer
        if values is None:
            values = np.linspace(-3, 3, n_steps)
        
        # Générer les images pour chaque valeur
        images = []
        vectors = []
        
        for value in values:
            # Copier le vecteur de base et modifier la dimension choisie
            latent_vector = base_vector.copy()
            latent_vector[dimension] = value
            vectors.append(latent_vector)
            
            # Générer l'image
            image = self.generate_from_vector(latent_vector)
            images.append(image[0])  # Extraire l'image du batch
        
        # Visualiser les résultats
        title = f"Exploration de la dimension {dimension}"
        fig_size = min(15, n_steps * 1.5)  # Limiter la taille maximale
        
        fig, axes = plt.subplots(1, len(images), figsize=(fig_size, 3))
        
        for i, (img, val) in enumerate(zip(images, values)):
            if len(images) > 1:
                ax = axes[i]
            else:
                ax = axes
                
            # Convertir l'image au format attendu par imshow
            if img.shape[0] == 1 or img.shape[0] == 3:  # Format [C, H, W]
                img = np.transpose(img, (1, 2, 0))
                
            if img.shape[-1] == 1:  # Image en niveaux de gris
                ax.imshow(img.squeeze(), cmap='gray')
            else:  # Image RGB
                ax.imshow(img)
                
            ax.set_title(f"Valeur: {val:.2f}")
            ax.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Visualisation sauvegardée: {save_path}")
            
        plt.show()
        
        return images, vectors
    
    def explore_principal_directions(self, n_components=10, n_steps=5, save_dir=None):
        """
        Identifie et explore les directions principales dans l'espace latent
        
        Args:
            n_components: Nombre de composantes principales à explorer
            n_steps: Nombre de pas dans chaque direction
            save_dir: Répertoire pour sauvegarder les visualisations
            
        Returns:
            Dictionnaire des composantes principales et images générées
        """
        from sklearn.decomposition import PCA
        
        # Créer un ensemble de vecteurs latents aléatoires
        n_samples = 1000
        logger.info(f"Génération de {n_samples} vecteurs latents aléatoires...")
        
        if self.is_tensorflow:
            latent_vectors = tf.random.normal([n_samples, self.latent_dim]).numpy()
        else:
            latent_vectors = torch.randn(n_samples, self.latent_dim).cpu().numpy()
        
        # Appliquer PCA pour trouver les directions principales
        logger.info("Calcul des composantes principales...")
        pca = PCA(n_components=min(n_components, self.latent_dim))
        pca.fit(latent_vectors)
        
        # Explorer chaque direction principale
        results = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        for i in range(min(n_components, self.latent_dim)):
            direction = pca.components_[i]
            variance = pca.explained_variance_ratio_[i] * 100
            
            logger.info(f"Exploration de la composante {i+1} "
                       f"(variance expliquée: {variance:.2f}%)...")
            
            # Vecteur moyen comme base
            mean_vector = np.zeros(self.latent_dim)
            
            # Définir les valeurs pour l'exploration
            values = np.linspace(-3, 3, n_steps)
            
            # Explorer cette direction
            images = []
            for value in values:
                latent_vector = mean_vector + direction * value
                img = self.generate_from_vector(latent_vector)
                images.append(img[0])
            
            # Sauvegarder la visualisation
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"pca_component_{i+1}.png")
                
            # Visualiser
            fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
            plt.suptitle(f"Composante principale {i+1} (variance: {variance:.2f}%)", 
                         fontsize=14)
            
            for j, (img, val) in enumerate(zip(images, values)):
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    axes[j].imshow(img.squeeze(), cmap='gray')
                else:
                    axes[j].imshow(img)
                    
                axes[j].set_title(f"Valeur: {val:.1f}")
                axes[j].axis('off')
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Visualisation sauvegardée: {save_path}")
                
            plt.show()
            
            results[i] = {
                'direction': direction,
                'variance': variance,
                'images': images
            }
            
        return results
    
    def interpolate_latent_vectors(self, start_vector, end_vector, n_steps=10, method='linear'):
        """
        Interpole entre deux vecteurs latents
        
        Args:
            start_vector: Vecteur latent de départ
            end_vector: Vecteur latent d'arrivée
            n_steps: Nombre d'étapes dans l'interpolation
            method: Méthode d'interpolation ('linear' ou 'spherical')
            
        Returns:
            Liste des vecteurs interpolés
        """
        # Convertir en numpy arrays si nécessaire
        if isinstance(start_vector, (tf.Tensor, torch.Tensor)):
            start_vector = start_vector.cpu().numpy() if isinstance(start_vector, torch.Tensor) else start_vector.numpy()
        if isinstance(end_vector, (tf.Tensor, torch.Tensor)):
            end_vector = end_vector.cpu().numpy() if isinstance(end_vector, torch.Tensor) else end_vector.numpy()
            
        # Assurer que les vecteurs sont 1D
        start_vector = start_vector.flatten()
        end_vector = end_vector.flatten()
        
        # Créer les interpolations
        vectors = []
        
        if method == 'linear':
            # Interpolation linéaire
            for t in np.linspace(0, 1, n_steps):
                vector = start_vector * (1 - t) + end_vector * t
                vectors.append(vector)
                
        elif method == 'spherical':
            # Interpolation sphérique (plus naturelle pour les GANs)
            start_norm = np.linalg.norm(start_vector)
            end_norm = np.linalg.norm(end_vector)
            
            start_dir = start_vector / start_norm
            end_dir = end_vector / end_norm
            
            # Calculer l'angle entre les deux vecteurs
            dot_product = np.dot(start_dir, end_dir)
            dot_product = min(max(dot_product, -1.0), 1.0)  # Assurer que c'est entre -1 et 1
            omega = np.arccos(dot_product)
            
            if omega < 1e-6:
                # Si les vecteurs sont presque identiques, utiliser l'interpolation linéaire
                for t in np.linspace(0, 1, n_steps):
                    vector = start_vector * (1 - t) + end_vector * t
                    vectors.append(vector)
            else:
                # Sinon, utiliser l'interpolation sphérique (slerp)
                for t in np.linspace(0, 1, n_steps):
                    # Calcul des coefficients
                    s0 = np.sin((1 - t) * omega) / np.sin(omega)
                    s1 = np.sin(t * omega) / np.sin(omega)
                    
                    # Interpolation des directions et des normes
                    direction = start_dir * s0 + end_dir * s1
                    norm = start_norm * (1 - t) + end_norm * t
                    
                    vector = direction * norm
                    vectors.append(vector)
        else:
            raise ValueError("Méthode d'interpolation non reconnue. Utilisez 'linear' ou 'spherical'.")
            
        return vectors
    
    def latent_space_interpolation(self, start_seed=None, end_seed=None, 
                                  n_steps=10, method='spherical', 
                                  animate=False, save_path=None):
        """
        Génère une interpolation entre deux points de l'espace latent
        
        Args:
            start_seed: Graine ou vecteur de départ (None pour aléatoire)
            end_seed: Graine ou vecteur d'arrivée (None pour aléatoire)
            n_steps: Nombre d'étapes dans l'interpolation
            method: Méthode d'interpolation ('linear' ou 'spherical')
            animate: Si True, crée une animation
            save_path: Chemin pour sauvegarder la visualisation ou l'animation
            
        Returns:
            Liste des images générées et des vecteurs interpolés
        """
        # Générer ou utiliser les vecteurs de départ et d'arrivée
        if start_seed is None:
            start_vector = self.random_latent_vector().numpy() if self.is_tensorflow else self.random_latent_vector().cpu().numpy()
        elif isinstance(start_seed, int):
            np.random.seed(start_seed)
            start_vector = np.random.normal(0, 1, self.latent_dim)
        else:
            start_vector = start_seed
            
        if end_seed is None:
            end_vector = self.random_latent_vector().numpy() if self.is_tensorflow else self.random_latent_vector().cpu().numpy()
        elif isinstance(end_seed, int):
            np.random.seed(end_seed)
            end_vector = np.random.normal(0, 1, self.latent_dim)
        else:
            end_vector = end_seed
            
        # Interpoler entre les vecteurs
        vectors = self.interpolate_latent_vectors(
            start_vector, end_vector, n_steps, method
        )
        
        # Générer les images pour chaque vecteur interpolé
        images = []
        for vector in vectors:
            img = self.generate_from_vector(vector)
            images.append(img[0])  # Prendre la première image du batch
            
        # Visualiser l'interpolation
        if animate:
            # Créer une animation
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.axis('off')
            
            plt.tight_layout()
            plt.close()
            
            def update(frame):
                ax.clear()
                ax.axis('off')
                
                img = images[frame]
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    ax.imshow(img.squeeze(), cmap='gray')
                else:
                    ax.imshow(img)
                    
                ax.set_title(f"Étape {frame+1}/{len(images)}")
                return [ax]
            
            # Créer et afficher l'animation
            ani = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)
            
            if save_path:
                ani.save(save_path, writer='pillow', fps=5)
                logger.info(f"Animation sauvegardée: {save_path}")
            
            # Afficher l'animation dans un notebook
            try:
                from IPython.display import HTML, display
                display(HTML(ani.to_jshtml()))
            except ImportError:
                plt.show()
        else:
            # Afficher une grille d'images
            fig_size = min(15, n_steps * 1.5)
            fig, axes = plt.subplots(1, len(images), figsize=(fig_size, 3))
            
            for i, img in enumerate(images):
                if len(images) > 1:
                    ax = axes[i]
                else:
                    ax = axes
                    
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    ax.imshow(img.squeeze(), cmap='gray')
                else:
                    ax.imshow(img)
                    
                ax.axis('off')
                
            plt.suptitle(f"Interpolation ({method})", fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Visualisation sauvegardée: {save_path}")
                
            plt.show()
            
        return images, vectors
    
    def create_latent_space_grid(self, dim1=0, dim2=1, n_steps=5, value_range=(-2, 2), 
                                base_vector=None, save_path=None):
        """
        Crée une grille d'images en variant deux dimensions de l'espace latent
        
        Args:
            dim1: Première dimension à varier (axe X)
            dim2: Deuxième dimension à varier (axe Y)
            n_steps: Nombre de pas dans chaque dimension
            value_range: Plage de valeurs pour les dimensions
            base_vector: Vecteur de base (None pour générer aléatoirement)
            save_path: Chemin pour sauvegarder la visualisation
            
        Returns:
            Grille d'images générées (liste 2D)
        """
        # Créer ou utiliser le vecteur de base
        if base_vector is None:
            base_vector = np.random.normal(0, 0.5, size=self.latent_dim)
            
        # Créer les valeurs pour les deux dimensions
        values = np.linspace(value_range[0], value_range[1], n_steps)
        
        # Générer la grille d'images
        grid = []
        
        for val2 in values:
            row = []
            for val1 in values:
                # Modifier les dimensions spécifiées
                vector = base_vector.copy()
                vector[dim1] = val1
                vector[dim2] = val2
                
                # Générer l'image
                img = self.generate_from_vector(vector)
                row.append(img[0])  # Prendre la première image du batch
                
            grid.append(row)
            
        # Visualiser la grille
        fig, axes = plt.subplots(n_steps, n_steps, figsize=(10, 10))
        
        for i, val2 in enumerate(values):
            for j, val1 in enumerate(values):
                img = grid[i][j]
                
                if img.shape[0] == 1 or img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                    
                if img.shape[-1] == 1:
                    axes[i, j].imshow(img.squeeze(), cmap='gray')
                else:
                    axes[i, j].imshow(img)
                    
                axes[i, j].axis('off')
                
                # Ajouter des labels aux axes
                if i == n_steps - 1:
                    axes[i, j].set_xlabel(f"{val1:.1f}")
                if j == 0:
                    axes[i, j].set_ylabel(f"{val2:.1f}")
                    
        # Ajouter des titres pour les axes
        fig.text(0.5, 0.02, f"Dimension {dim1}", ha='center', fontsize=12)
        fig.text(0.02, 0.5, f"Dimension {dim2}", va='center', rotation='vertical', fontsize=12)
        plt.suptitle(f"Variation des dimensions {dim1} et {dim2} de l'espace latent", fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Grille sauvegardée: {save_path}")
            
        plt.show()
        
        return grid