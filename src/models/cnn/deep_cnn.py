import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.framework_utils import FrameworkBridge
import logging
import os
from datetime import datetime

# Réduction de la verbosité des logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DeepCNN:
    def __init__(self, input_shape=(64, 64, 3), latent_dim=100):
        """
        Initialize a Deep CNN GAN for image generation
        
        Args:
            input_shape: Shape of the target images (height, width, channels)
            latent_dim: Dimension of the latent space (noise vector size)
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build both generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Create combined model (generator + discriminator) for training the generator
        self.combined = self.build_combined_model()
        
        # Pour suivi des métriques
        self.metrics_history = {
            'd_loss': [], 'd_acc': [], 
            'g_loss': [], 
            'epoch_d_loss': [], 'epoch_g_loss': [],
            'epoch_d_acc': []
        }
        
    def build_generator(self):
        """Build a lightweight generator network"""
        noise_shape = (self.latent_dim,)
        
        model = models.Sequential(name="Generator_Lite")
        
        # First dense layer - réduit de 256 à 128 filtres
        model.add(layers.Dense(4 * 4 * 128, input_shape=noise_shape))
        model.add(layers.Reshape((4, 4, 128)))
        
        # Couches de convolution transposées - 3 couches au lieu de 4
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Dernière couche de upsampling - correspond maintenant à 32x32 au lieu de 64x64
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Output layer with tanh activation
        model.add(layers.Conv2D(self.input_shape[2], (3, 3), padding='same', activation='tanh'))
        
        # S'assurer que toutes les couches sont bien trainable
        for layer in model.layers:
            layer.trainable = True
            
        return model
        
    def build_discriminator(self):
        """Build a lightweight discriminator network"""
        model = models.Sequential(name="Discriminator_Lite")
        
        # Couches de convolution réduites - 3 couches au lieu de 4
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', 
                             input_shape=self.input_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))  # Réduit le dropout
        
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # S'assurer que toutes les couches sont bien trainable
        for layer in model.layers:
            layer.trainable = True
            
        # Compile avec la même configuration
        model.compile(loss='binary_crossentropy',
                     optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     metrics=['accuracy'])
                     
        return model
        
    def build_combined_model(self):
        """Build combined model for training the generator"""
        # Pour le modèle combiné, on fige temporairement le discriminateur
        self.discriminator.trainable = False
        
        # Create combined model (generator + discriminator)
        z_input = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z_input)
        validity = self.discriminator(img)
        
        # Combined model takes noise as input and outputs validity
        combined = models.Model(z_input, validity)
        combined.compile(loss='binary_crossentropy',
                       optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
        
        return combined
    
    def preprocess_batch_data(self, batch_data):
        """Version optimisée avec mise en cache"""
        batch_key = id(batch_data)  # Identifiant unique pour ce batch
        
        # Si ce batch est déjà dans le cache, retourner la version traitée
        if hasattr(self, '_batch_cache') and batch_key in self._batch_cache:
            return self._batch_cache[batch_key]
        
        # Créer le dictionnaire de cache si nécessaire
        if not hasattr(self, '_batch_cache'):
            self._batch_cache = {}
            
        # Traitement normal du batch
        # Gérer les deux types de retour possibles du dataloader
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            imgs = batch_data[0]  # Loader retourne (images, labels)
        else:
            imgs = batch_data  # Loader retourne uniquement les images
        
        # Conversion des données via FrameworkBridge amélioré
        try:
            # 1. Conversion en tensor TensorFlow
            if FrameworkBridge.is_pytorch_tensor(imgs):
                imgs = FrameworkBridge.pytorch_to_tensorflow(imgs, normalize_range=(-1, 1))
            elif not FrameworkBridge.is_tensorflow_tensor(imgs):
                # Si ce n'est ni PyTorch ni TensorFlow, convertir via numpy
                array = FrameworkBridge.to_numpy(imgs)
                imgs = tf.convert_to_tensor(array, dtype=tf.float32)
            
            # 2. Normalisation de la forme du tensor
            imgs = FrameworkBridge.normalize_batch_shape(imgs, expected_format='NHWC')
            
            # 3. Vérification finale de compatibilité avec le modèle
            expected_shape = (None, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            actual_shape = imgs.shape
            
            if len(actual_shape) != 4 or actual_shape[3] != expected_shape[3]:
                raise ValueError(f"Forme du tensor incompatible: attendu {expected_shape}, obtenu {actual_shape}")
            
            # Stocker dans le cache (limiter à 5 entrées pour éviter les fuites mémoire)
            if len(self._batch_cache) > 5:
                # Supprimer l'entrée la plus ancienne
                oldest_key = next(iter(self._batch_cache))
                del self._batch_cache[oldest_key]
            
            self._batch_cache[batch_key] = imgs
            return imgs
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {e}")
            raise
    
    def train(self, train_loader, epochs=20, log_interval=50, plot_interval=500, save_interval=5):
        """Train the GAN model"""
        # Vérifier le dataloader
        if train_loader is None or len(train_loader) == 0:
            logger.warning("Dataloader vide, entraînement impossible")
            return self.metrics_history
        
        # Créer un optimiseur dédié pour le générateur 
        # (puisque nous n'utilisons pas celui du modèle combiné avec GradientTape)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Répertoire pour les checkpoints
        checkpoint_dir = os.path.join("checkpoints", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(epochs):
            d_losses_epoch = []
            g_losses_epoch = []
            d_accs_epoch = []
            
            for i, batch in enumerate(train_loader):
                # Prétraitement des données une seule fois
                imgs = self.preprocess_batch_data(batch)
                current_batch_size = imgs.shape[0]
                
                # Créer les labels pour ce batch
                real = np.ones((current_batch_size, 1))
                fake = np.zeros((current_batch_size, 1))
                
                # ----------------------
                # IMPORTANT: S'assurer que le discriminateur est trainable
                # ----------------------
                self.discriminator.trainable = True
                
                # Générer des images fausses
                noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise, verbose=0)
                
                # Entraîner le discriminateur
                d_loss_real = self.discriminator.train_on_batch(imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ----------------------
                # Entraîner le générateur SOIT avec le modèle combiné SOIT avec GradientTape
                # MAIS pas les deux (ce qui cause des conflits)
                # ----------------------
                
                # OPTION 1: Utiliser le modèle combiné (plus simple)
                noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, real)
                
                # OPTION 2: Utiliser GradientTape (nous commentons cette partie)
                """
                # Figer le discriminateur pour l'entraînement du générateur
                self.discriminator.trainable = False
                
                # Générer du bruit frais pour le générateur
                noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
                
                # Utiliser TensorFlow GradientTape pour un meilleur suivi des gradients
                with tf.GradientTape() as tape:
                    gen_imgs = self.generator(noise, training=True)
                    fake_outputs = self.discriminator(gen_imgs, training=False)
                    g_loss = tf.keras.losses.binary_crossentropy(real, fake_outputs)
                
                # Calculer et appliquer les gradients manuellement
                grads = tape.gradient(g_loss, self.generator.trainable_weights)
                generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
                
                g_loss = tf.reduce_mean(g_loss).numpy()
                """
                
                # Enregistrer les pertes
                self.metrics_history["d_loss"].append(d_loss[0])
                self.metrics_history["d_acc"].append(d_loss[1])
                self.metrics_history["g_loss"].append(g_loss)
                
                d_losses_epoch.append(d_loss[0])
                d_accs_epoch.append(d_loss[1])
                g_losses_epoch.append(g_loss)
                
                # Vérifier l'équilibre GAN
                if i % log_interval == 0:
                    gan_balance = d_loss[0] / (g_loss + 1e-8)  # éviter division par zéro
                    balance_status = "équilibré" if 0.5 <= gan_balance <= 2 else "déséquilibré"
                    
                    logger.warning(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(train_loader)}] "
                          f"[D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] "
                          f"[G loss: {g_loss:.4f}] [Balance GAN: {balance_status}]")
                
                # Générer et sauvegarder des images à certains intervalles
                if i % plot_interval == 0:
                    self.save_sample_images(epoch)
            
            # À la fin de chaque époque
            avg_d_loss = sum(d_losses_epoch) / len(d_losses_epoch) if d_losses_epoch else 0
            avg_g_loss = sum(g_losses_epoch) / len(g_losses_epoch) if g_losses_epoch else 0
            avg_d_acc = sum(d_accs_epoch) / len(d_accs_epoch) if d_accs_epoch else 0
            
            self.metrics_history["epoch_d_loss"].append(avg_d_loss)
            self.metrics_history["epoch_g_loss"].append(avg_g_loss)
            self.metrics_history["epoch_d_acc"].append(avg_d_acc)
            
            logger.warning(f"Epoch {epoch+1}/{epochs} - "
                  f"Avg D loss: {avg_d_loss:.4f}, Avg D acc: {100*avg_d_acc:.2f}%, "
                  f"Avg G loss: {avg_g_loss:.4f}")
            
            # Vérification de l'apprentissage
            if epoch > 0:
                d_progress = self.metrics_history["epoch_d_loss"][-2] - avg_d_loss
                g_progress = self.metrics_history["epoch_g_loss"][-2] - avg_g_loss
                
                if abs(d_progress) < 0.001 and abs(g_progress) < 0.001:
                    logger.warning("Attention: L'apprentissage semble stagner!")
            
            # Sauvegarder le modèle périodiquement
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"gan_checkpoint_epoch_{epoch+1}")
                self.save_model(checkpoint_path)
                    
        # Enregistrer le modèle final
        final_path = os.path.join(checkpoint_dir, "gan_model_final")
        self.save_model(final_path)
        return self.metrics_history
    
    def generate_images(self, num_images=1):
        """Generate new images from random noise
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            Array of generated images with values in [0, 1]
        """
        # Generate random noise
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        
        # Generate images from noise
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        return gen_imgs
    
    def save_sample_images(self, epoch):
        """Save sample generated images to visualize training progress
        
        Args:
            epoch: Current training epoch
        """
        # Créer le dossier si nécessaire
        os.makedirs("images", exist_ok=True)
    
        r, c = 4, 4  # Grid size
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
                
        fig.savefig(f"images/gan_generated_epoch_{epoch}.png")
        plt.close()
    
    def save_model(self, filepath):
        """Sauvegarde les modèles générateur et discriminateur
        
        Args:
            filepath: Chemin de base pour la sauvegarde des modèles
        """
        # Créer le répertoire parent si nécessaire
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Utiliser le format moderne .keras si disponible
        try:
            # Tester si la méthode moderne de sauvegarde est disponible
            generator_path = f"{filepath}_generator.keras"
            discriminator_path = f"{filepath}_discriminator.keras"
            self.generator.save(generator_path)
            self.discriminator.save(discriminator_path)
            logger.info(f"Modèles sauvegardés au format .keras: {generator_path}, {discriminator_path}")
        except (ValueError, ImportError):
            # Fallback sur le format h5 legacy
            generator_path = f"{filepath}_generator.h5"
            discriminator_path = f"{filepath}_discriminator.h5"
            self.generator.save(generator_path)
            self.discriminator.save(discriminator_path)
            logger.info(f"Modèles sauvegardés au format .h5: {generator_path}, {discriminator_path}")
            
        return filepath
        
    def load_model(self, model_path):
        """
        Charge un modèle GAN précédemment enregistré
        
        Args:
            model_path: Chemin de base des fichiers sauvegardés (sans _generator.h5/_discriminator.h5)
        
        Returns:
            self: L'instance du modèle avec les poids chargés
        """
        generator_path = f"{model_path}_generator.h5"
        discriminator_path = f"{model_path}_discriminator.h5"
        
        # Vérifier que les fichiers existent
        if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
            raise FileNotFoundError(f"Fichiers de modèle non trouvés: {generator_path} ou {discriminator_path}")
        
        try:
            # Charger d'abord le générateur pour analyser sa configuration
            self.generator = tf.keras.models.load_model(generator_path)
            logger.info(f"Générateur chargé depuis {generator_path}")
            
            # Détecter automatiquement la dimension latente utilisée dans le modèle sauvegardé
            # MÉTHODE AMÉLIORÉE: analyser directement les couches du générateur
            detected_latent_dim = None
            
            # Option 1: Rechercher une couche InputLayer explicite
            for layer in self.generator.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    input_shape = layer.input_shape
                    if input_shape and len(input_shape) == 2:
                        detected_latent_dim = input_shape[1]
                        logger.info(f"Dimension latente détectée dans InputLayer: {detected_latent_dim}")
                        break
            
            # Option 2: Si le générateur est un modèle séquentiel, vérifier la première couche
            if detected_latent_dim is None and isinstance(self.generator, tf.keras.Sequential):
                first_layer = self.generator.layers[0] 
                if hasattr(first_layer, 'input_shape'):
                    input_specs = getattr(first_layer, 'input_spec', None)
                    if input_specs is not None and hasattr(input_specs, 'axes'):
                        # Extraire la dimension d'entrée des spécifications d'axes
                        axes = input_specs.axes
                        if axes and -1 in axes:
                            detected_latent_dim = axes[-1]
                            logger.info(f"Dimension latente détectée dans les specs d'entrée: {detected_latent_dim}")
                            
                    # Alternative: utiliser l'attribut batch_input_shape de la première couche
                    elif hasattr(first_layer, 'batch_input_shape'):
                        input_shape = first_layer.batch_input_shape
                        if input_shape and len(input_shape) == 2:
                            detected_latent_dim = input_shape[1]
                            logger.info(f"Dimension latente détectée dans batch_input_shape: {detected_latent_dim}")
            
            # Option 3: Dernier recours - essayer d'inférer à partir des couches internes
            if detected_latent_dim is None:
                # Regarder les poids de la première couche Dense si applicable
                for layer in self.generator.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        weights = layer.get_weights()
                        if weights and len(weights) > 0:
                            w = weights[0]  # Premier élément est la matrice de poids
                            detected_latent_dim = w.shape[0]  # Forme d'entrée
                            logger.info(f"Dimension latente inférée des poids: {detected_latent_dim}")
                            break
            
            # Mettre à jour la dimension latente si détectée
            if detected_latent_dim is not None:
                if detected_latent_dim != self.latent_dim:
                    logger.warning(f"Dimension latente détectée ({detected_latent_dim}) différente "
                                  f"de celle spécifiée ({self.latent_dim}). Utilisation de la valeur détectée.")
                    self.latent_dim = detected_latent_dim
            else:
                logger.warning(f"Impossible de détecter la dimension latente. Utilisation de la valeur par défaut: {self.latent_dim}")
            
            # Charger le discriminateur
            self.discriminator = tf.keras.models.load_model(discriminator_path)
            logger.info(f"Discriminateur chargé depuis {discriminator_path}")
            
            # IMPORTANT: Reconstruire le modèle combiné avec la dimension latente CORRECTE
            # Au lieu d'utiliser build_combined_model, on implémente directement ici
            self.discriminator.trainable = False
            
            # Créer un modèle combiné sur mesure avec la dimension latente correcte
            z_input = tf.keras.layers.Input(shape=(self.latent_dim,))
            try:
                # Version 1: Utiliser call() pour la conversion des données
                img = self.generator(z_input)
                validity = self.discriminator(img)
                self.combined = tf.keras.models.Model(z_input, validity)
            except Exception as e:
                # Version 2: Si la première méthode échoue, essayer une approche alternative
                logger.warning(f"Erreur lors de la première tentative de création du modèle combiné: {e}")
                logger.warning("Tentative de reconstruction avec méthode alternative...")
                
                # Créer un modèle séquentiel pour le générateur gv2
                generator_v2 = tf.keras.models.clone_model(self.generator)
                generator_v2.set_weights(self.generator.get_weights())
                
                # Créer un nouveau modèle complet
                z_input = tf.keras.layers.Input(shape=(self.latent_dim,))
                x = generator_v2(z_input)
                validity = self.discriminator(x)
                
                self.combined = tf.keras.models.Model(z_input, validity)
                
            # Compiler le modèle combiné
            self.combined.compile(loss='binary_crossentropy',
                                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
            
            logger.info(f"Modèle combiné reconstruit avec dimension latente = {self.latent_dim}")
            
            # Vérifier que le modèle fonctionne avec un test simple
            try:
                test_noise = tf.random.normal([2, self.latent_dim])
                test_out = self.combined(test_noise)
                logger.info(f"Test du modèle combiné réussi: sortie de forme {test_out.shape}")
            except Exception as e:
                logger.error(f"Test du modèle combiné échoué: {str(e)}")
                raise ValueError(f"Le modèle combiné ne fonctionne pas correctement: {str(e)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def evaluate(self, data_source=None, n_samples=1000, output_dir="results/evaluations"):
        """
        Effectue une évaluation complète du modèle GAN
        
        Args:
            data_source: Source des données réelles (dataloader ou nom du dataset)
            n_samples: Nombre d'échantillons à utiliser pour l'évaluation
            output_dir: Répertoire de sortie pour les résultats
            
        Returns:
            Dictionnaire contenant les résultats de l'évaluation
        """
        from src.evaluation.gan_evaluator import GANEvaluator
        
        # Charger les données réelles si nécessaire
        if isinstance(data_source, str):
            from src.data.data_loader import get_gan_data_loader
            data_loader = get_gan_data_loader(
                data_source, 
                batch_size=32, 
                image_size=self.input_shape[0]
            )
        else:
            data_loader = data_source
        
        # Créer l'évaluateur
        evaluator = GANEvaluator(
            model=self,
            data_loader=data_loader,
            output_dir=output_dir
        )
        
        # Exécuter l'évaluation
        results = evaluator.run_full_evaluation(n_samples=n_samples)
        
        return results

    def generate_images(self, num_images=16, noise=None):
        """
        Génère des images à partir du générateur
        
        Args:
            num_images: Nombre d'images à générer
            noise: Bruit à utiliser (None pour générer un bruit aléatoire)
            
        Returns:
            Images générées sous forme de tensor ou array numpy
        """
        try:
            # Générer du bruit aléatoire si non fourni
            if noise is None:
                noise = np.random.normal(0, 1, (num_images, self.latent_dim))
                
            # Générer les images
            images = self.generator.predict(noise, verbose=0)
            
            # Normaliser dans [0, 1] si nécessaire
            if np.min(images) < 0:
                images = (images + 1) / 2.0
                
            return images
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'images: {e}")
            raise