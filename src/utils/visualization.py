import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import torchvision.utils as vutils
from PIL import Image
import matplotlib.gridspec as gridspec

def plot_generated_images(images, n_cols=5, title=None, save_path=None):
    """Affiche une grille d'images générées
    
    Args:
        images: Liste ou tenseur d'images à afficher
        n_cols: Nombre de colonnes dans la grille
        title: Titre optionnel de la figure
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    # Convertir en tenseur si ce n'est pas déjà le cas
    if not isinstance(images, torch.Tensor):
        if isinstance(images, list) and isinstance(images[0], np.ndarray):
            # Convertir la liste de numpy arrays en tenseur
            images = torch.from_numpy(np.stack(images))
        else:
            images = torch.tensor(images)
    
    # Assurer que les valeurs sont entre [0, 1] pour l'affichage
    if images.min() < 0 or images.max() > 1:
        images = (images + 1) / 2 if images.min() < 0 else images / 255.0
    
    # S'assurer que le format est correct: [batch_size, channels, height, width]
    if images.dim() == 3 and (images.size(0) == 1 or images.size(0) == 3):
        # C'est une seule image avec format [channels, height, width]
        images = images.unsqueeze(0)
    elif images.dim() == 3:
        # Format [batch_size, height, width] (images en niveaux de gris)
        images = images.unsqueeze(1)
    
    n_images = images.size(0)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    grid = vutils.make_grid(images, nrow=n_cols, normalize=False, padding=2)
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.tight_layout()
    plt.show()

def compare_real_generated(real_images, generated_images, n_samples=5, title=None, save_path=None):
    """Compare des images réelles avec des images générées
    
    Args:
        real_images: Lot d'images réelles (tenseur ou liste)
        generated_images: Lot d'images générées (tenseur ou liste)
        n_samples: Nombre d'exemples à afficher
        title: Titre optionnel pour la figure
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    # Convertir en tenseurs si nécessaire
    if not isinstance(real_images, torch.Tensor):
        real_images = torch.tensor(real_images)
    if not isinstance(generated_images, torch.Tensor):
        generated_images = torch.tensor(generated_images)
    
    # Normaliser entre [0, 1] si nécessaire
    if real_images.min() < 0 or real_images.max() > 1:
        real_images = (real_images + 1) / 2 if real_images.min() < 0 else real_images / 255.0
    if generated_images.min() < 0 or generated_images.max() > 1:
        generated_images = (generated_images + 1) / 2 if generated_images.min() < 0 else generated_images / 255.0
        
    # Assurer format correct
    if real_images.dim() == 3 and (real_images.size(0) == 1 or real_images.size(0) == 3):
        real_images = real_images.unsqueeze(0)
    if generated_images.dim() == 3 and (generated_images.size(0) == 1 or generated_images.size(0) == 3):
        generated_images = generated_images.unsqueeze(0)
    
    # Sélectionner jusqu'à n_samples images
    n_samples = min(n_samples, real_images.size(0), generated_images.size(0))
    real_images = real_images[:n_samples]
    generated_images = generated_images[:n_samples]
    
    # Créer la figure
    fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Afficher les images réelles sur la première ligne
    for i in range(n_samples):
        img = real_images[i].cpu()
        if img.dim() == 4:  # Si c'est un batch de taille 1
            img = img.squeeze(0)
        if img.size(0) == 1:  # Image en niveaux de gris
            axs[0, i].imshow(img.squeeze(0).numpy(), cmap='gray')
        else:  # Image RGB
            axs[0, i].imshow(img.permute(1, 2, 0).numpy())
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title("Images réelles")
    
    # Afficher les images générées sur la deuxième ligne
    for i in range(n_samples):
        img = generated_images[i].cpu()
        if img.dim() == 4:  # Si c'est un batch de taille 1
            img = img.squeeze(0)
        if img.size(0) == 1:  # Image en niveaux de gris
            axs[1, i].imshow(img.squeeze(0).numpy(), cmap='gray')
        else:  # Image RGB
            axs[1, i].imshow(img.permute(1, 2, 0).numpy())
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title("Images générées")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

def plot_generated_texts(texts, prompts=None, title="Textes générés", save_path=None):
    """Affiche les textes générés par un modèle
    
    Args:
        texts: Liste de textes générés
        prompts: Liste optionnelle des prompts utilisés
        title: Titre de la visualisation
        save_path: Chemin pour sauvegarder le résultat (optionnel)
    """
    if not texts:
        print("Aucun texte fourni pour la visualisation.")
        return
    
    # Créer un affichage formaté
    result = f"# {title}\n\n"
    
    for i, text in enumerate(texts):
        if prompts and i < len(prompts):
            result += f"### Exemple {i+1} - Prompt: \"{prompts[i]}\"\n\n"
        else:
            result += f"### Exemple {i+1}\n\n"
        
        result += f"```\n{text.strip()}\n```\n\n"
    
    # Afficher le résultat
    print(result)
    
    # Sauvegarder si demandé
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Résultat sauvegardé dans {save_path}")

def plot_diffusion_process(images, title="Processus de diffusion", save_path=None, animated=True):
    """Visualise le processus de diffusion (débruitage progressif)
    
    Args:
        images: Liste de tenseurs représentant les étapes du processus de diffusion
                Format attendu: liste de tenseurs de forme [C, H, W] ou [B, C, H, W]
        title: Titre de la visualisation
        save_path: Chemin pour sauvegarder l'animation (optionnel)
        animated: Si True, créer une animation, sinon afficher une grille d'images
    """
    # Vérifier et préparer les images
    prepared_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # Normaliser si nécessaire
            if img.min() < 0 or img.max() > 1:
                img = (img + 1) / 2 if img.min() < 0 else img / 255.0
            # Assurer le format correct
            if img.dim() == 4:  # Format [B, C, H, W]
                img = img[0]  # Prendre la première image du batch
            img = img.cpu().detach().numpy()
        prepared_images.append(img)
    
    if animated:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Fonction d'initialisation pour l'animation
        def init():
            if prepared_images[0].shape[0] == 1:  # Image en niveaux de gris
                ax.imshow(prepared_images[0].squeeze(0), cmap='gray')
            else:  # Image RGB
                ax.imshow(np.transpose(prepared_images[0], (1, 2, 0)))
            return [ax]
        
        # Fonction d'animation
        def animate(i):
            ax.clear()
            ax.axis('off')
            if prepared_images[i].shape[0] == 1:  # Image en niveaux de gris
                ax.imshow(prepared_images[i].squeeze(0), cmap='gray')
            else:  # Image RGB
                ax.imshow(np.transpose(prepared_images[i], (1, 2, 0)))
            ax.set_title(f'Étape {i+1}/{len(prepared_images)}')
            return [ax]
        
        # Créer l'animation
        anim = FuncAnimation(fig, animate, frames=len(prepared_images),
                             init_func=init, interval=200, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        
        # Afficher l'animation
        plt.close()
        display(HTML(anim.to_jshtml()))
        
    else:
        # Afficher en grille
        n_images = len(prepared_images)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        for i, img in enumerate(prepared_images):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            if img.shape[0] == 1:  # Image en niveaux de gris
                ax.imshow(img.squeeze(0), cmap='gray')
            else:  # Image RGB
                ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(f'Étape {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()

def plot_attention_weights(attention_weights, tokens=None, title="Poids d'attention", save_path=None):
    """Visualise la matrice des poids d'attention des modèles Transformer
    
    Args:
        attention_weights: Matrice des poids d'attention [N, N]
        tokens: Liste optionnelle des tokens à afficher sur les axes
        title: Titre de la visualisation
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().detach().numpy()
    
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Afficher la heatmap
    cax = ax.matshow(attention_weights, cmap="viridis")
    fig.colorbar(cax)
    
    if title:
        plt.title(title)
    
    # Ajouter les labels des tokens si fournis
    if tokens:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

def plot_training_history(history, metrics=None, save_path=None):
    """Affiche l'historique d'entraînement pour un ou plusieurs modèles
    
    Args:
        history: Dictionnaire contenant les métriques d'entraînement, ou liste de dictionnaires
                pour comparer plusieurs modèles
        metrics: Liste des métriques à afficher (si None, affiche tout)
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    if not isinstance(history, list):
        history = [history]
    
    # Déterminer les métriques à afficher
    if metrics is None:
        all_metrics = set()
        for hist in history:
            all_metrics.update(hist.keys())
        metrics = list(all_metrics)
    
    # Nombre de sous-graphiques
    n_plots = len(metrics)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, metric in enumerate(metrics):
        plt.subplot(n_rows, n_cols, i + 1)
        
        for j, hist in enumerate(history):
            if metric in hist:
                label = f"Modèle {j+1}" if len(history) > 1 else metric
                plt.plot(hist[metric], label=label)
        
        plt.title(metric)
        plt.xlabel('Époque')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()