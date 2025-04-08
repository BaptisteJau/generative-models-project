import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tqdm import tqdm
import logging
from datetime import datetime
from PIL import Image
import seaborn as sns

from src.utils.metrics import InceptionScore, FrechetInceptionDistance
from src.utils.framework_utils import FrameworkBridge
from src.utils.visualization import compare_real_generated
from src.data.data_loader import get_gan_data_loader

logger = logging.getLogger(__name__)

class GANEvaluator:
    """Évalue en profondeur les performances d'un modèle GAN"""
    
    def __init__(self, model, dataset=None, data_loader=None, device=None,
                 output_dir="results/evaluations", batch_size=32):
        """
        Args:
            model: Modèle GAN à évaluer (DeepCNN)
            dataset: Dataset contenant des images réelles (optionnel)
            data_loader: DataLoader pour les images réelles (optionnel)
            device: Appareil sur lequel exécuter l'évaluation ('cuda', 'cpu')
            output_dir: Répertoire pour sauvegarder les résultats
            batch_size: Taille des batchs pour la génération d'images
        """
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = os.path.join(output_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.batch_size = batch_size
        
        # Créer le répertoire de sortie
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        
        # Initialiser les métriques
        self.inception_score = InceptionScore(device=self.device)
        self.fid = FrechetInceptionDistance(device=self.device)
        
        self.results = {}
    
    def generate_evaluation_samples(self, n_samples=1000, save_samples=True):
        """Génère des échantillons pour l'évaluation
        
        Args:
            n_samples: Nombre d'échantillons à générer
            save_samples: Si True, sauvegarde des échantillons représentatifs
            
        Returns:
            Tensor d'images générées [N, C, H, W] (format PyTorch)
        """
        logger.info(f"Génération de {n_samples} échantillons pour évaluation...")
        
        # Batchs pour la génération
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        generated_samples = []
        
        for i in tqdm(range(n_batches), desc="Génération d'échantillons"):
            # Calculer la taille du batch (le dernier peut être plus petit)
            current_batch_size = min(self.batch_size, n_samples - i * self.batch_size)
            
            # Générer des images
            images = self.model.generate_images(num_images=current_batch_size)
            
            # Si TensorFlow, convertir en PyTorch pour l'évaluation
            if isinstance(images, (tf.Tensor, np.ndarray)):
                images = torch.tensor(images.astype(np.float32))
                
            # Format PyTorch: [B, C, H, W]
            if images.shape[1] != 3 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            
            generated_samples.append(images)
            
        # Concaténer tous les échantillons
        all_samples = torch.cat(generated_samples, dim=0)
        
        # Sauvegarder quelques échantillons
        if save_samples:
            # Sélectionner 16 échantillons aléatoires
            indices = np.random.choice(len(all_samples), min(16, len(all_samples)), replace=False)
            sample_grid = torch.stack([all_samples[i] for i in indices])
            
            # Sauvegarder la grille
            self.save_image_grid(sample_grid, os.path.join(self.output_dir, "samples", "generated_samples.png"))
        
        return all_samples
    
    def load_real_samples(self, n_samples=1000, data_source=None):
        """Charge des échantillons réels pour la comparaison
        
        Args:
            n_samples: Nombre d'échantillons à charger
            data_source: Source des données (si différente de self.dataset)
            
        Returns:
            Tensor d'images réelles [N, C, H, W] (format PyTorch)
        """
        # Utiliser le data_loader fourni s'il existe
        if self.data_loader is not None:
            return self._load_from_dataloader(self.data_loader, n_samples)
            
        # Utiliser le dataset si fourni
        if self.dataset is not None:
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            return self._load_from_dataloader(loader, n_samples)
        
        # Utiliser une source spécifiée
        if data_source is not None:
            if isinstance(data_source, str):  # Chemin ou nom de dataset
                data_loader = get_gan_data_loader(data_source, batch_size=self.batch_size)
                return self._load_from_dataloader(data_loader, n_samples)
            else:  # Supposer que c'est un DataLoader
                return self._load_from_dataloader(data_source, n_samples)
        
        raise ValueError("Aucune source de données réelles fournie.")
    
    def _load_from_dataloader(self, loader, n_samples):
        """Charge des échantillons à partir d'un DataLoader"""
        real_samples = []
        samples_loaded = 0
        
        for batch in tqdm(loader, desc="Chargement d'échantillons réels"):
            # Si le batch est un tuple/liste (images, labels), prendre uniquement les images
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            # Si le batch est en format NCHW, aucun changement nécessaire
            # Si en NHWC, convertir en NCHW pour PyTorch
            if images.dim() == 4 and images.shape[1] != 3 and images.shape[3] == 3:
                images = images.permute(0, 3, 1, 2)
            
            real_samples.append(images)
            
            samples_loaded += images.shape[0]
            if samples_loaded >= n_samples:
                break
        
        all_samples = torch.cat(real_samples, dim=0)[:n_samples]
        return all_samples
    
    def calculate_metrics(self, generated_images, real_images):
        """Calcule les métriques quantitatives (FID, IS)
        
        Args:
            generated_images: Images générées en format PyTorch [N, C, H, W]
            real_images: Images réelles en format PyTorch [N, C, H, W]
            
        Returns:
            Dict avec les métriques calculées
        """
        metrics = {}
        
        # Redimensionner les images pour Inception (299×299)
        real_resized = self._resize_for_inception(real_images)
        gen_resized = self._resize_for_inception(generated_images)
        
        # Calcul du score Inception
        logger.info("Calcul du score Inception...")
        is_mean, is_std = self.inception_score(gen_resized)
        metrics['inception_score'] = is_mean
        metrics['inception_score_std'] = is_std
        
        # Calcul de la distance FID
        logger.info("Calcul de la distance FID...")
        fid_score = self.fid(real_resized, gen_resized)
        metrics['fid'] = fid_score
        
        self.results.update(metrics)
        return metrics
    
    def _resize_for_inception(self, images, target_size=299):
        """Redimensionne les images au format attendu par Inception"""
        import torch.nn.functional as F
        
        # Si les images sont déjà à la bonne taille
        if images.shape[2] == target_size and images.shape[3] == target_size:
            return images
            
        return F.interpolate(images, size=(target_size, target_size), 
                            mode='bilinear', align_corners=False)
    
    def analyze_diversity(self, images, n_clusters=5):
        """Analyse la diversité des images générées en utilisant un clustering
        
        Args:
            images: Tensor d'images générées [N, C, H, W]
            n_clusters: Nombre de clusters à former
            
        Returns:
            Score de diversité et visualisation des clusters
        """
        from sklearn.cluster import KMeans
        import torch.nn as nn
        
        logger.info("Analyse de la diversité des images générées...")
        
        # Utiliser un modèle pré-entraîné pour extraire des caractéristiques
        extractor = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 
                                                      'resnet18', pretrained=True).children())[:-1])
        extractor.eval().to(self.device)
        
        # Extraire les caractéristiques (vecteurs de 512 dimensions)
        features = []
        batch_size = 64  # Plus petit batch pour éviter les OOM
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch_features = extractor(batch).squeeze(-1).squeeze(-1).cpu().numpy()
                features.append(batch_features)
        
        features = np.vstack(features)
        
        # Clustering K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(features)
        
        # Calcul d'un score de diversité basé sur la dispersion des clusters
        diversity_score = kmeans.inertia_ / len(images)  # Inertie normalisée
        
        # Visualiser des exemples de chaque cluster
        cluster_samples = {}
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                # Prendre jusqu'à 4 exemples de ce cluster
                sample_indices = np.random.choice(cluster_indices, 
                                                min(4, len(cluster_indices)), 
                                                replace=False)
                cluster_samples[i] = [images[idx] for idx in sample_indices]
        
        # Créer une visualisation des clusters
        self.visualize_clusters(cluster_samples, 
                               os.path.join(self.output_dir, "diversity_clusters.png"))
        
        # Enregistrer le résultat
        self.results['diversity_score'] = diversity_score
        
        return diversity_score
    
    def visualize_clusters(self, cluster_samples, save_path):
        """Visualise des exemples de chaque cluster"""
        n_clusters = len(cluster_samples)
        n_samples = max(len(samples) for samples in cluster_samples.values())
        
        fig, axs = plt.subplots(n_clusters, n_samples, 
                               figsize=(n_samples * 3, n_clusters * 3))
        
        for i, (cluster_idx, samples) in enumerate(cluster_samples.items()):
            for j, img in enumerate(samples):
                img_np = img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                if n_clusters > 1:
                    axs[i, j].imshow(img_np)
                    axs[i, j].set_title(f"Cluster {cluster_idx+1}")
                    axs[i, j].axis('off')
                else:
                    axs[j].imshow(img_np)
                    axs[j].set_title(f"Cluster {cluster_idx+1}")
                    axs[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    
    def visualize_learning_curves(self):
        """Visualise les courbes d'apprentissage à partir de l'historique du modèle"""
        if not hasattr(self.model, 'metrics_history') or not self.model.metrics_history:
            logger.warning("Pas d'historique d'apprentissage disponible.")
            return
        
        metrics = self.model.metrics_history
        
        # Créer un répertoire pour les courbes
        curves_dir = os.path.join(self.output_dir, "learning_curves")
        os.makedirs(curves_dir, exist_ok=True)
        
        # Tracer les pertes du discriminateur et du générateur
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(metrics['epoch_d_loss'])+1)
        
        ax.plot(epochs, metrics['epoch_d_loss'], 'b-', label='Discriminator Loss')
        ax.plot(epochs, metrics['epoch_g_loss'], 'r-', label='Generator Loss')
        ax.set_title('Évolution des pertes pendant l\'entraînement')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter une ligne horizontale à y=0.5 pour la référence
        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "losses.png"))
        plt.close(fig)
        
        # Tracer la précision du discriminateur
        if 'epoch_d_acc' in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, metrics['epoch_d_acc'], 'g-', label='Discriminator Accuracy')
            ax.set_title('Précision du discriminateur pendant l\'entraînement')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "accuracy.png"))
            plt.close(fig)
        
        # Équilibre GAN (ratio D_loss/G_loss)
        fig, ax = plt.subplots(figsize=(10, 5))
        gan_balance = [d/(g+1e-8) for d, g in zip(metrics['epoch_d_loss'], metrics['epoch_g_loss'])]
        ax.plot(epochs, gan_balance, 'purple', label='GAN Balance (D/G ratio)')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect balance')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Generator advantage')
        ax.axhline(y=2.0, color='blue', linestyle='--', alpha=0.5, label='Discriminator advantage')
        ax.set_title('Équilibre GAN pendant l\'entraînement')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('D_loss / G_loss')
        ax.set_ylim(0, 5)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, "gan_balance.png"))
        plt.close(fig)
    
    def save_image_grid(self, images, save_path):
        """Sauvegarde une grille d'images"""
        import torchvision
        
        # Créer une grille
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=True)
        
        # Convertir en image PIL et sauvegarder
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        Image.fromarray(grid).save(save_path)
    
    def compare_with_real_samples(self, generated_images, real_images, n_samples=5):
        """Compare des échantillons générés avec des échantillons réels"""
        # Sélectionner aléatoirement n_samples
        gen_indices = np.random.choice(len(generated_images), n_samples, replace=False)
        real_indices = np.random.choice(len(real_images), n_samples, replace=False)
        
        gen_subset = torch.stack([generated_images[i] for i in gen_indices])
        real_subset = torch.stack([real_images[i] for i in real_indices])
        
        # Utiliser la fonction existante pour comparer
        compare_real_generated(
            real_subset, 
            gen_subset, 
            n_samples=n_samples,
            title="Comparaison des échantillons réels et générés",
            save_path=os.path.join(self.output_dir, "real_vs_generated.png")
        )
    
    def analyze_quality_distribution(self, generated_images):
        """Analyse la distribution de la qualité des images générées"""
        from torch.nn import functional as F
        import torchvision.models as models
        
        # Charger un classificateur pré-entrainé pour évaluer la qualité
        model = models.resnet18(pretrained=True).to(self.device)
        model.eval()
        
        # Fonction pour calculer un score de qualité (basé sur l'entropie des prédictions)
        softmax = torch.nn.Softmax(dim=1)
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(generated_images), self.batch_size):
                batch = generated_images[i:i+self.batch_size].to(self.device)
                predictions = softmax(model(batch))
                
                # La confiance de la classe la plus probable comme score de qualité
                quality_scores = torch.max(predictions, dim=1)[0].cpu().numpy()
                scores.extend(quality_scores)
        
        # Visualiser la distribution des scores de qualité
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, kde=True)
        plt.title("Distribution des scores de qualité des images générées")
        plt.xlabel("Score de qualité")
        plt.ylabel("Fréquence")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "quality_distribution.png"))
        plt.close()
        
        # Sauvegarder des exemples de différentes qualités
        indices = np.argsort(scores)
        
        # Sélectionner 5 des pires et 5 des meilleures images
        worst_indices = indices[:5]
        best_indices = indices[-5:]
        
        worst_samples = torch.stack([generated_images[i] for i in worst_indices])
        best_samples = torch.stack([generated_images[i] for i in best_indices])
        
        # Sauvegarder les grilles
        self.save_image_grid(worst_samples, os.path.join(self.output_dir, "worst_samples.png"))
        self.save_image_grid(best_samples, os.path.join(self.output_dir, "best_samples.png"))
        
        self.results['quality_mean'] = np.mean(scores)
        self.results['quality_std'] = np.std(scores)
        
        return scores
    
    def run_full_evaluation(self, n_samples=1000, data_source=None):
        """Exécute l'ensemble du pipeline d'évaluation
        
        Args:
            n_samples: Nombre d'échantillons à utiliser
            data_source: Source des données réelles (optionnel)
            
        Returns:
            Dict avec tous les résultats d'évaluation
        """
        logger.info("Démarrage de l'évaluation complète du GAN...")
        
        # 1. Générer des échantillons
        generated_samples = self.generate_evaluation_samples(n_samples)
        
        # 2. Charger des échantillons réels
        real_samples = self.load_real_samples(n_samples, data_source)
        
        # 3. Comparer quelques échantillons réels et générés
        self.compare_with_real_samples(generated_samples, real_samples)
        
        # 4. Calculer les métriques de qualité
        metrics = self.calculate_metrics(generated_samples, real_samples)
        logger.info(f"Métriques calculées: Inception Score = {metrics['inception_score']:.3f}, FID = {metrics['fid']:.3f}")
        
        # 5. Analyser la diversité
        diversity_score = self.analyze_diversity(generated_samples)
        logger.info(f"Score de diversité: {diversity_score:.5f}")
        
        # 6. Analyser la distribution de qualité
        quality_scores = self.analyze_quality_distribution(generated_samples)
        logger.info(f"Qualité moyenne: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")
        
        # 7. Visualiser les courbes d'apprentissage
        self.visualize_learning_curves()
        
        # 8. Générer un rapport
        self.generate_report()
        
        return self.results
    
    # Correction de la méthode generate_report pour échapper correctement les accolades CSS

    def generate_report(self):
        """Génère un rapport HTML avec tous les résultats"""
        report_path = os.path.join(self.output_dir, "evaluation_report.html")
        
        # IMPORTANT: Doubler les accolades pour les échapper dans les règles CSS
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GAN Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
                .metric {{ font-weight: bold; }}
                img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>GAN Evaluation Report</h1>
            <p>Generated on: {date}</p>
            
            <div class="section">
                <h2>Quantitative Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Ajouter les métriques
        for key, value in self.results.items():
            if isinstance(value, (int, float)):
                html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value:.4f}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Generated Samples</h2>
                <img src="samples/generated_samples.png" alt="Generated Samples">
            </div>
            
            <div class="section">
                <h2>Real vs Generated Comparison</h2>
                <img src="real_vs_generated.png" alt="Real vs Generated">
            </div>
            
            <div class="section">
                <h2>Learning Curves</h2>
                <img src="learning_curves/losses.png" alt="Learning Curves">
                <img src="learning_curves/accuracy.png" alt="Discriminator Accuracy">
                <img src="learning_curves/gan_balance.png" alt="GAN Balance">
            </div>
            
            <div class="section">
                <h2>Diversity Analysis</h2>
                <img src="diversity_clusters.png" alt="Diversity Clusters">
                <p>Diversity Score: <span class="metric">{diversity:.5f}</span></p>
            </div>
            
            <div class="section">
                <h2>Quality Distribution</h2>
                <img src="quality_distribution.png" alt="Quality Distribution">
                <p>Mean Quality: <span class="metric">{quality_mean:.3f} ± {quality_std:.3f}</span></p>
                
                <h3>Best Samples</h3>
                <img src="best_samples.png" alt="Best Samples">
                
                <h3>Worst Samples</h3>
                <img src="worst_samples.png" alt="Worst Samples">
            </div>
        </body>
        </html>
        """.format(
            diversity=self.results.get('diversity_score', 0),
            quality_mean=self.results.get('quality_mean', 0),
            quality_std=self.results.get('quality_std', 0)
        )
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Rapport d'évaluation généré: {report_path}")
        return report_path