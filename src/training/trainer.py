import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

class BaseTrainer:
    """Classe de base pour tous les trainers"""
    def __init__(self, model, train_loader, val_loader=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = self.config.get('num_epochs', 10)
        self.save_dir = self.config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        """Méthode à implémenter dans les classes dérivées"""
        raise NotImplementedError("Subclass must implement abstract method")
    
    def validate(self):
        """Valider le modèle Transformer"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extraire input_ids et labels
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Format tuple (inputs, targets)
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    attention_mask = None
                else:
                    raise ValueError("Transformer validation requires batches as dictionary with 'input_ids'/'labels' or as tuple (inputs, targets)")
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Traitement similaire à la méthode train()
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.size(-1))
                    labels = labels.reshape(-1)
                
                # Calculer la perte
                loss = self.criterion(logits, labels)
                
                val_loss += loss.item()
                num_batches += 1
        
        # Moyenne de perte pour cette validation
        avg_val_loss = val_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, perplexity
    
    def save_model(self, epoch):
        """Sauvegarde du modèle"""
        os.makedirs(self.save_dir, exist_ok=True)
        model_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        return model_path


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
                args['epochs'] = self.epochs
            
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
            
            for epoch in range(self.epochs):
                d_losses = []
                g_losses = []
                d_accs = []
                
                # Barre de progression
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
                
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
                
                print(f"Epoch {epoch+1}/{self.epochs} - D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, D Acc: {epoch_d_acc:.2f}")
                
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
            return self.model.train(self.train_loader, epochs=self.epochs)
        
        # Implémentation manuelle pour les modèles sans méthode train
        else:
            history = {"loss": []}
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
            # Boucle d'entraînement
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Barre de progression
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
                
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
                
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
                
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
        super().__init__(model, train_loader, val_loader, config)
        
        # Use optimizer from model or create a new one
        self.optimizer = getattr(model, 'optimizer', None)
        if self.optimizer is None:
            lr = config.get('learning_rate', 0.001)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr
            )
        
        self.criterion = nn.CrossEntropyLoss()
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def train(self):
        """Entraîner le modèle Transformer pour la génération de texte"""
        history = {"train_loss": [], "val_loss": [], "perplexity": []}
        
        # Déplacer le modèle vers le device approprié
        self.model = self.model.to(self.device)
        
        for epoch in range(self.epochs):
            # Entraînement
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Barre de progression
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in pbar:
                # Extraire input_ids et labels
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    # Ignorer attention_mask car notre modèle ne le supporte pas
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    raise ValueError("Transformer training requires batches as dictionary with 'input_ids'/'labels' or as tuple (inputs, targets)")
                
                # Forward pass - utiliser seulement les input_ids
                try:
                    # Essayer avec la signature qui semble correspondre à votre modèle
                    outputs = self.model(input_ids, trg=labels)
                except TypeError as e:
                    try:
                        # Essayer d'autres signatures communes
                        outputs = self.model(input_ids, labels)
                    except TypeError:
                        # Si tout échoue, afficher un message d'erreur utile
                        raise TypeError(f"Impossible d'appeler le modèle Transformer. Vérifiez les arguments attendus par la méthode forward(): {e}")
                
                # Reformatter si nécessaire pour calculer la perte
                if hasattr(outputs, 'logits'):
                    # Pour les modèles HuggingFace
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    # Pour des sorties standard (batch_size, seq_len, vocab_size)
                    logits = outputs
                
                if logits.dim() > 2:
                    # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
                    logits = logits.reshape(-1, logits.size(-1))
                    # [batch_size, seq_len] -> [batch_size * seq_len]
                    labels = labels.reshape(-1)
                
                # Calculer la perte
                loss = self.criterion(logits, labels)
                
                # Optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Enregistrer la perte
                train_loss += loss.item()
                num_batches += 1
                
                # Mettre à jour la barre de progression
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Moyenne de perte pour cette époque
            avg_train_loss = train_loss / num_batches
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if self.val_loader:
                val_loss, perplexity = self.validate()
                history["val_loss"].append(val_loss)
                history["perplexity"].append(perplexity)
                
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Sauvegarder le modèle périodiquement
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_model(epoch)
                # Générer un exemple de texte
                self.generate_text_sample(epoch)
        
        return history
    
    def validate(self):
        """Valider le modèle Transformer"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extraire input_ids et labels
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    # Ignorer attention_mask car notre modèle ne le supporte pas
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    # Format tuple (inputs, targets)
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    raise ValueError("Transformer validation requires batches as dictionary with 'input_ids'/'labels' or as tuple (inputs, targets)")
                
                # Forward pass - utiliser la même méthode que dans train()
                try:
                    # Essayer avec la signature qui correspond à votre modèle
                    outputs = self.model(input_ids, trg=labels)
                except TypeError as e:
                    try:
                        # Essayer d'autres signatures communes
                        outputs = self.model(input_ids, labels)
                    except TypeError:
                        # Si tout échoue, afficher un message d'erreur utile
                        raise TypeError(f"Impossible d'appeler le modèle Transformer. Vérifiez les arguments attendus par la méthode forward(): {e}")
                
                # Reformatter si nécessaire pour calculer la perte
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                if logits.dim() > 2:
                    logits = logits.reshape(-1, logits.size(-1))
                    labels = labels.reshape(-1)
                
                # Calculer la perte
                loss = self.criterion(logits, labels)
                
                val_loss += loss.item()
                num_batches += 1
        
        # Moyenne de perte pour cette validation
        avg_val_loss = val_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, perplexity
    
    def generate_text_sample(self, epoch):
        """Génère un exemple de texte"""
        # Assurez-vous que le modèle est en mode évaluation
        self.model.eval()
        
        # Vérifier si le modèle a une méthode generate
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            # Différents prompts pour tester la génération
            prompts = [
                "Once upon a time",
                "The meaning of life is",
                "In the beginning",
                "The future of AI"
            ]
            
            sample_file = os.path.join(self.save_dir, f"text_samples_epoch_{epoch+1}.txt")
            
            # Inspecter les paramètres attendus par la méthode generate
            import inspect
            generate_params = inspect.signature(self.model.generate).parameters
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    try:
                        # Tokeniser le prompt
                        if hasattr(self.model, 'tokenizer'):
                            tokenizer = self.model.tokenizer
                        else:
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained('gpt2')
                        
                        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
                        
                        # Générer du texte avec les bons paramètres
                        with torch.no_grad():
                            # Utiliser les paramètres appropriés selon l'inspection
                            kwargs = {}
                            
                            # Gérer différentes possibilités pour le paramètre de longueur
                            if 'max_length' in generate_params:
                                kwargs['max_length'] = 100
                            elif 'max_new_tokens' in generate_params:
                                kwargs['max_new_tokens'] = 100
                            elif 'length' in generate_params:
                                kwargs['length'] = 100
                            
                            # Si la méthode accepte temperature
                            if 'temperature' in generate_params:
                                kwargs['temperature'] = 0.8
                                
                            # Si la méthode accepte des input_ids
                            if 'input_ids' in generate_params:
                                output = self.model.generate(input_ids=input_ids, **kwargs)
                            else:
                                # Sinon, passer directement le prompt
                                output = self.model.generate(prompt, **kwargs)
                        
                        # Décoder et écrire le texte généré
                        if isinstance(output, torch.Tensor):
                            if hasattr(self.model, 'tokenizer'):
                                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                            else:
                                generated_text = f"[Tensor output not decodable: {output.shape}]"
                        else:
                            # Si le modèle renvoie déjà du texte
                            generated_text = output
                        
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Generated: {generated_text}\n\n")
                    
                    except Exception as e:
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Error generating text: {str(e)}\n\n")
                        print(f"Error generating text for prompt '{prompt}': {str(e)}")


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