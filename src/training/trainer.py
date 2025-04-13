import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import contextlib  # Ajout de l'import manquant
from tqdm import tqdm
import torchvision.utils as vutils
import logging
from datetime import datetime
from src.utils.logging_config import configure_logging
from src.utils.early_stopping import EarlyStopping
from contextlib import nullcontext  # Ajouter cet import

# Configurer le logger pour ce module
logger = logging.getLogger(__name__)

class BaseTrainer:
    """Classe de base pour tous les trainers"""
    
    def __init__(self, model, train_loader, val_loader=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Attributs communs
        self.start_epoch = 1
        self.num_epochs = self.config.get('num_epochs', 10)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Répertoire de sauvegarde
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(
            self.config.get('save_dir', 'checkpoints'), 
            f"{self.__class__.__name__}_{timestamp}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        """Méthode principale d'entraînement à implémenter dans les sous-classes"""
        raise NotImplementedError("Les sous-classes doivent implémenter train()")
        
    def validate(self):
        """Méthode de validation à implémenter dans les sous-classes"""
        raise NotImplementedError("Les sous-classes doivent implémenter validate()")
        
    def save_checkpoint(self, epoch, is_best=False, final=False):
        """Sauvegarde un checkpoint du modèle"""
        try:
            checkpoint_dir = self.save_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if final:
                checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
            elif is_best:
                checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            else:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            
            logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du checkpoint: {e}")


class GANTrainer(BaseTrainer):
    """Trainer spécialisé pour les modèles GAN (CNN génératif)"""
    
    def __init__(self, model, train_loader, val_loader=None, config=None):
        super().__init__(model, train_loader, val_loader, config)
        # Paramètres spécifiques aux GANs
        self.latent_dim = self.config.get('latent_dim', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.save_interval = self.config.get('save_interval', 100)
        self.image_dir = self.config.get('image_dir', 'generated_images')
        os.makedirs(self.image_dir, exist_ok=True)
        
    def train(self):
        """Entraînement du GAN avec alternance generator/discriminator"""
        # Pour TensorFlow DeepCNN
        if hasattr(self.model, 'train'):
            # Le modèle DeepCNN a sa propre méthode train
            # Vérifions quels paramètres sont acceptés par la méthode train du modèle
            import inspect
            train_params = inspect.signature(self.model.train).parameters
            
            # Construisons un dictionnaire avec seulement les paramètres supportés
            args = {'train_loader': self.train_loader}
            
            if 'epochs' in train_params:
                args['epochs'] = self.num_epochs  # CORRECTION: self.epochs → self.num_epochs
            
            if 'log_interval' in train_params:
                args['log_interval'] = self.config.get('log_interval', 10)
            
            if 'plot_interval' in train_params:
                args['plot_interval'] = self.config.get('save_interval', 100)
                
            # Appeler la méthode train avec les bons arguments
            return self.model.train(**args)
            
        # Pour PyTorch GAN
        else:
            # Labels pour images réelles (1) et fausses (0)
            real_label = 1
            fake_label = 0
            
            # Historique d'entraînement
            history = {"d_loss": [], "g_loss": [], "d_acc": []}
            
            for epoch in range(self.num_epochs):  # CORRECTION: self.epochs → self.num_epochs
                d_losses = []
                g_losses = []
                d_accs = []
                
                # Barre de progression
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
                
                for real_images in pbar:
                    # Déplacer sur le device approprié
                    if isinstance(real_images, (list, tuple)):
                        real_images = real_images[0]  # Pour les loaders qui retournent (images, _)
                    real_images = real_images.to(self.device)
                    batch_size = real_images.size(0)
                    
                    # Labels
                    real_targets = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                    fake_targets = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
                    
                    # -----------------
                    # Entraîner le discriminateur
                    # -----------------
                    self.model.discriminator.zero_grad()
                    
                    # Format pour le discriminateur
                    if real_images.dim() == 3:
                        real_images = real_images.unsqueeze(1)  # Ajouter canal pour grayscale
                    
                    # Prédiction sur images réelles
                    real_output = self.model.discriminator(real_images)
                    d_loss_real = nn.BCELoss()(real_output, real_targets)
                    d_loss_real.backward()
                    
                    # Générer des fausses images
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_images = self.model.generator(noise)
                    
                    # Prédiction sur images fausses
                    fake_output = self.model.discriminator(fake_images.detach())
                    d_loss_fake = nn.BCELoss()(fake_output, fake_targets)
                    d_loss_fake.backward()
                    
                    # Mettre à jour le discriminateur
                    d_loss = d_loss_real + d_loss_fake
                    self.model.d_optimizer.step()
                    
                    # Calculer précision du discriminateur
                    pred_real = (real_output > 0.5).float()
                    pred_fake = (fake_output <= 0.5).float()
                    d_acc = (torch.sum(pred_real) + torch.sum(pred_fake)) / (2 * batch_size)
                    
                    # -----------------
                    # Entraîner le générateur
                    # -----------------
                    self.model.generator.zero_grad()
                    
                    # Le générateur veut que le discriminateur se trompe
                    output = self.model.discriminator(fake_images)
                    g_loss = nn.BCELoss()(output, real_targets)  # Tromper le discriminateur
                    g_loss.backward()
                    self.model.g_optimizer.step()
                    
                    # Enregistrer les pertes
                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())
                    d_accs.append(d_acc.item())
                    
                    # Mettre à jour la barre de progression
                    pbar.set_postfix({
                        "D Loss": f"{d_loss.item():.4f}",
                        "G Loss": f"{g_loss.item():.4f}",
                        "D Acc": f"{d_acc.item():.2f}"
                    })
                
                # Enregistrer les moyennes pour cette époque
                epoch_d_loss = sum(d_losses) / len(d_losses)
                epoch_g_loss = sum(g_losses) / len(g_losses)
                epoch_d_acc = sum(d_accs) / len(d_accs)
                
                history['d_loss'].append(epoch_d_loss)
                history['g_loss'].append(epoch_g_loss)
                history['d_acc'].append(epoch_d_acc)
                
                print(f"Epoch {epoch+1}/{self.num_epochs} - D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, D Acc: {epoch_d_acc:.2f}")
                
                # Sauvegarder des exemples générés
                if (epoch + 1) % self.save_interval == 0:
                    self.save_generated_images(epoch)
                    self.save_model(epoch)
            
            return history
    
    def validate(self):
        """Génère des images et calcule des métriques de qualité"""
        # Pour GANs, la validation consiste généralement à générer des images
        # et à calculer des métriques comme Inception Score ou FID
        # Ici, nous implémentons simplement la génération d'images
        with torch.no_grad():
            noise = torch.randn(64, self.latent_dim, device=self.device)
            generated_images = self.model.generator(noise)
            return generated_images
    
    def save_generated_images(self, epoch):
        """Sauvegarde des images générées pendant l'entraînement"""
        with torch.no_grad():
            noise = torch.randn(64, self.latent_dim, device=self.device)
            generated_images = self.model.generator(noise)
            
            # Normaliser les images pour l'affichage (de [-1, 1] à [0, 1])
            generated_images = (generated_images + 1) / 2
            
            # Créer une grille d'images
            grid = vutils.make_grid(generated_images[:16], nrow=4, normalize=False)
            
            # Sauvegarder l'image
            image_path = os.path.join(self.image_dir, f"epoch_{epoch+1}.png")
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Epoch {epoch+1}")
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig(image_path)
            plt.close()
    
    def save_model(self, epoch):
        """Sauvegarde les modèles à un point donné de l'entraînement"""
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Générer nom de fichier
        checkpoint_path = os.path.join(checkpoint_dir, f"gan_checkpoint_epoch_{epoch+1}")
        
        # Détecter si le modèle a sa propre méthode de sauvegarde
        if hasattr(self.model, 'save_model') and callable(getattr(self.model, 'save_model')):
            self.model.save_model(checkpoint_path)
            logger.info(f"Checkpoint GAN sauvegardé à l'epoch {epoch+1}")
        else:
            # Si c'est un modèle PyTorch standard
            gen_path = f"{checkpoint_path}_generator.pt"
            disc_path = f"{checkpoint_path}_discriminator.pt"
            
            try:
                torch.save(self.model.generator.state_dict(), gen_path)
                torch.save(self.model.discriminator.state_dict(), disc_path)
                logger.info(f"Checkpoint GAN PyTorch sauvegardé à l'epoch {epoch+1}")
            except AttributeError:
                logger.warning(f"Impossible de sauvegarder le checkpoint à l'epoch {epoch+1}")


class DiffusionTrainer(BaseTrainer):
    """Trainer spécialisé pour les modèles de diffusion"""
    
    def __init__(self, model, train_loader, val_loader=None, config=None):
        super().__init__(model, train_loader, val_loader, config)
        self.sample_dir = self.config.get('sample_dir', 'diffusion_samples')
        os.makedirs(self.sample_dir, exist_ok=True)
        self.sample_interval = self.config.get('sample_interval', 10)
        
    def train(self):
        """Entraîner le modèle de diffusion"""
        # Pour les modèles qui ont leur propre méthode d'entraînement
        if hasattr(self.model, 'train') and callable(getattr(self.model, 'train')):
            return self.model.train(self.train_loader, epochs=self.num_epochs)
        
        # Implémentation manuelle pour les modèles sans méthode train
        else:
            history = {"loss": []}
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
            # Boucle d'entraînement
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Barre de progression
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
                
                for batch in pbar:
                    # Extraire les images
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                    
                    images = images.to(self.device)
                    batch_size = images.shape[0]
                    
                    # Normaliser si nécessaire
                    if images.min() >= 0 and images.max() <= 1:
                        images = 2 * images - 1  # Scale to [-1, 1]
                    
                    # Choisir des timesteps aléatoires
                    t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
                    
                    # Ajouter du bruit selon le timestep
                    noise = torch.randn_like(images)
                    noisy_images = self.model.add_noise(images, t, noise)
                    
                    # Prédire le bruit
                    predicted_noise = self.model(noisy_images, t)
                    
                    # Calculer la perte
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                    
                    # Optimisation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Enregistrer la perte
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Mettre à jour la barre de progression
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Moyenne de perte pour cette époque
                avg_loss = epoch_loss / num_batches
                history["loss"].append(avg_loss)
                
                print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}")
                
                # Générer des échantillons périodiquement
                if (epoch + 1) % self.sample_interval == 0:
                    self.generate_samples(epoch)
                    self.save_model(epoch)
            
            return history
    
    def generate_samples(self, epoch):
        """Génère des échantillons à partir du modèle de diffusion"""
        num_samples = 4
        
        # Si le modèle a sa propre méthode de génération
        if hasattr(self.model, 'generate_samples') and callable(getattr(self.model, 'generate_samples')):
            samples = self.model.generate_samples(
                n=num_samples, 
                save_path=os.path.join(self.sample_dir, f"epoch_{epoch+1}.png")
            )
        else:
            with torch.no_grad():
                # Commencer par du bruit
                x = torch.randn(num_samples, 3, 64, 64, device=self.device)
                
                # Débruitage progressif
                for t in tqdm(reversed(range(self.model.num_timesteps)), desc="Sampling"):
                    x = self.model.denoise_step(x, t)
                
                # Normaliser pour affichage
                samples = (x + 1) / 2  # De [-1, 1] à [0, 1]
                samples = torch.clamp(samples, 0, 1)
                
                # Sauvegarder les échantillons
                grid = vutils.make_grid(samples, nrow=2, normalize=False)
                image_path = os.path.join(self.sample_dir, f"epoch_{epoch+1}.png")
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title(f"Epoch {epoch+1}")
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.savefig(image_path)
                plt.close()
        
        return samples
    
    def validate(self):
        """Pour les modèles de diffusion, valide en générant des échantillons"""
        return self.generate_samples("validation")


class TransformerTrainer(BaseTrainer):
    """Trainer spécialisé pour les modèles Transformer génératifs"""
    
    def __init__(self, model, train_loader, val_loader=None, config=None):
        """
        Initialise le gestionnaire d'entraînement pour les Transformers
        
        Args:
            model: Le modèle Transformer à entraîner
            train_loader: DataLoader pour les données d'entraînement
            val_loader: DataLoader pour les données de validation
            config: Dictionnaire de configuration pour l'entraînement
        """
        super().__init__(model, train_loader, val_loader, config)
        
        # Extraire save_every de la configuration
        self.save_every = self.config.get('save_every', 5)  # Ajout de cette ligne
        
        # Optimizer
        lr = config.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Logging
        logger.info(f"TransformerTrainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Ajouter le scheduler de taux d'apprentissage si activé
        self.use_scheduler = config.get('use_scheduler', False)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2, 
                verbose=True
            )
        
        # Configuration pour l'early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 0)  # 0 désactive l'early stopping
    
    def train(self):
        """Entraîner le modèle Transformer"""
        logger.info(f"Début de l'entraînement sur {self.device}")
        
        # Configuration de l'accumulation de gradient
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        logger.info(f"Accumulation de gradients: {gradient_accumulation_steps} étapes")
        
        # Configuration de la précision mixte
        use_amp = self.config.get('use_amp', False)
        scaler = torch.amp.GradScaler() if use_amp else None
        logger.info(f"Précision mixte: {'activée' if use_amp else 'désactivée'}")
        
        # Configuration de l'early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        # Meilleur modèle et son epoch
        best_val_loss = float('inf')
        best_epoch = -1
        
        # Initialisation des compteurs
        patience_counter = 0
        
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda') if use_amp else nullcontext():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient accumulation
                if use_amp:
                    scaler.scale(loss).backward()
                    if (batch_count + 1) % gradient_accumulation_steps == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_count + 1) % gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1
                
                pbar.set_postfix(loss=loss.item())
            
            # Calculer les pertes moyennes
            avg_train_loss = total_loss / batch_count
            
            # Évaluation
            val_loss, perplexity = self.validate()
            
            # Appliquer le scheduler si activé
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            # Early stopping si activé
            if self.early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Sauvegarder le meilleur modèle
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping activé à l'époque {epoch}")
                    break
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}, Val perplexity: {perplexity:.2f}")
            
            # Sauvegarde périodique
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch)
                self.generate_text_sample(epoch)
                
            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
        
        # Fin de l'entraînement
        logger.info(f"Entraînement terminé. Meilleur modèle à l'époque {best_epoch+1} avec perte de validation = {best_val_loss:.4f}")
        
        # Sauvegarder une dernière fois
        self.save_checkpoint(self.num_epochs-1, final=True)
        self.generate_text_sample(self.num_epochs-1)
        
        # Restaurer le meilleur modèle pour la dernière génération
        best_path = os.path.join(self.save_dir, f"best_checkpoint.pt")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path)['model_state_dict'])
            self.generate_text_sample("best")

        return {
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'perplexity': perplexity
        }
        
    def validate(self):
        """Valider le modèle Transformer"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        # Obtenir la taille du vocabulaire depuis le dataset
        vocab_size = None
        if hasattr(self.train_loader.dataset, 'dataset') and hasattr(self.train_loader.dataset.dataset, 'vocab_size'):
            vocab_size = self.train_loader.dataset.dataset.vocab_size
        elif hasattr(self.train_loader.dataset, 'vocab_size'):
            vocab_size = self.train_loader.dataset.vocab_size
        
        # Si toujours pas trouvé, essayer de l'obtenir du modèle
        if vocab_size is None and hasattr(self.model, 'embedding'):
            vocab_size = self.model.embedding.num_embeddings
            
        if vocab_size is None:
            raise ValueError("Impossible de déterminer la taille du vocabulaire")
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extraire input_ids et labels
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    raise ValueError("Format de batch non supporté pour validation")
                
                # Forward pass - SANS utiliser l'argument 'labels'
                outputs = self.model(input_ids)
                
                # Calculer la perte avec la taille du vocabulaire récupérée
                loss = self.criterion(outputs.view(-1, vocab_size), labels.view(-1))
                
                val_loss += loss.item()
                num_batches += 1
                
                # Libérer la mémoire GPU
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Moyenne de perte pour cette validation
        avg_val_loss = val_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, perplexity
    
    def generate_text_sample(self, epoch):
        """Génère un exemple de texte"""
        try:
            self.model.eval()
            
            # Récupérer le tokenizer depuis le dataloader
            tokenizer = None
            if hasattr(self.train_loader, 'dataset'):
                if hasattr(self.train_loader.dataset, 'tokenizer'):
                    tokenizer = self.train_loader.dataset.tokenizer
                elif hasattr(self.train_loader.dataset, 'dataset') and hasattr(self.train_loader.dataset.dataset, 'tokenizer'):
                    tokenizer = self.train_loader.dataset.dataset.tokenizer
            
            # Attacher le tokenizer au modèle si disponible
            if tokenizer and not hasattr(self.model, 'tokenizer'):
                self.model.tokenizer = tokenizer
            
            prompts = ["Once upon a time", "The future of AI", "In a galaxy far"]
            sample_file = os.path.join(self.save_dir, f"text_samples_epoch_{epoch}.txt")
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    try:
                        # Utiliser une température dynamique (plus élevée au début de l'entraînement)
                        # pour favoriser l'exploration, puis plus basse pour la cohérence
                        temperature = 1.3  # Température élevée pour éviter les répétitions
                        
                        generated_text = self.model.generate(
                            prompt, 
                            max_length=100,
                            temperature=temperature,
                            top_k=40,
                            top_p=0.92,
                            repetition_penalty=1.8  # Pénalité élevée pour éviter les répétitions
                        )
                        
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Generated: {generated_text}\n\n")
                    except Exception as e:
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Error generating text: {str(e)}\n\n")
                        logger.error(f"Erreur lors de la génération de texte: {e}")
            
            logger.info(f"Exemples de texte générés et sauvegardés dans {sample_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'exemples de texte: {e}")


class CNNTrainer(BaseTrainer):
    def train(self):
        """Train the CNN model (GAN)"""
        # Obtenir les paramètres d'entraînement du config
        epochs = self.config.get('epochs', 100)
        log_interval = self.config.get('log_interval', 10)
        plot_interval = self.config.get('plot_interval', 100)
        
        # Appeler la méthode train du modèle sans passer batch_size
        return self.model.train(
            self.train_loader,
            epochs=epochs,
            log_interval=log_interval,
            plot_interval=plot_interval
        )


# Fonction factory pour créer le bon trainer selon le type de modèle
def create_trainer(model_type, model, train_loader, val_loader=None, config=None):
    """
    Crée le trainer approprié selon le type de modèle
    
    Args:
        model_type: Type de modèle ('cnn', 'transformer', 'diffusion')
        model: Instance du modèle
        train_loader: DataLoader pour l'entraînement
        val_loader: DataLoader pour la validation (optionnel)
        config: Configuration d'entraînement
        
    Returns:
        Une instance du trainer approprié
    """
    if model_type.lower() in ['cnn', 'gan', 'dcgan']:
        return GANTrainer(model, train_loader, val_loader, config)
    elif model_type.lower() in ['transformer']:
        return TransformerTrainer(model, train_loader, val_loader, config)
    elif model_type.lower() in ['diffusion']:
        return DiffusionTrainer(model, train_loader, val_loader, config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")