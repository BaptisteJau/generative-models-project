import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import os
from datetime import datetime

# Configuration du logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class SpectralNormalization(tf.keras.layers.Layer):
    """Normalisation spectrale pour les couches convolutionnelles"""
    def __init__(self, layer, power_iterations=1, **kwargs):
        self.power_iterations = power_iterations
        self.layer = layer
        super(SpectralNormalization, self).__init__(**kwargs)
            
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
            
        # Initialisation pour la puissance itérative
        self.u = self.add_weight(
            name='u',
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False
        )
        super(SpectralNormalization, self).build(input_shape)
            
    def call(self, inputs):
        self._update_weights()
        output = self.layer(inputs)
        return output
            
    def _update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
            
        # Approximation de la norme spectrale
        for _ in range(self.power_iterations):
            v = tf.matmul(self.u, w_reshaped, transpose_b=True)
            v = tf.nn.l2_normalize(v)
            u = tf.matmul(v, w_reshaped)
            u = tf.nn.l2_normalize(u)
            
        # Normalisation des poids
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        self.u.assign(u)
        self.layer.kernel.assign(self.w / sigma)

class DeepCNN:
    def __init__(self, input_shape=(64, 64, 3), latent_dim=100, use_spectral_norm=False, use_wasserstein=False):
        """
        Initialisation d'un GAN CNN avancé avec normalisation spectrale et Wasserstein
        
        Args:
            input_shape: Forme des images cibles (hauteur, largeur, canaux)
            latent_dim: Dimension de l'espace latent (taille du vecteur de bruit)
            use_spectral_norm: Utiliser la normalisation spectrale dans le discriminateur
            use_wasserstein: Utiliser la perte Wasserstein au lieu de BCE
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.use_spectral_norm = use_spectral_norm
        self.use_wasserstein = use_wasserstein
        
        # Créer le générateur et le discriminateur
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Créer le modèle combiné
        self.combined = self.build_combined_model()
        
        # Historique des métriques
        self.metrics_history = {
            'd_loss': [], 'd_acc': [], 
            'g_loss': [], 
            'epoch_d_loss': [], 'epoch_g_loss': [],
            'epoch_d_acc': []
        }
        
        # Cache pour optimisation du traitement des données
        self._batch_cache = {}
        
    def build_generator(self):
        """Générateur avec connexions résiduelles et anti-effondrement"""
        z_input = layers.Input(shape=(self.latent_dim,))
        
        # Projection initiale et reshape
        x = layers.Dense(8 * 8 * 512, use_bias=False)(z_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((8, 8, 512))(x)
        
        # Blocs de upsampling avec connexions résiduelles
        x = self._generator_block(x, 256, upsample=True)  # 8x8 -> 16x16
        x = self._generator_block(x, 128, upsample=True)  # 16x16 -> 32x32
        
        # Bloc supplémentaire pour 64x64 si nécessaire
        if self.input_shape[0] == 64:
            x = self._generator_block(x, 64, upsample=True)  # 32x32 -> 64x64
        
        # Couche de sortie avec activation tanh
        output = layers.Conv2D(self.input_shape[2], kernel_size=3, padding='same', 
                              activation='tanh', use_bias=False)(x)
        
        model = models.Model(z_input, output, name="Generator_Enhanced")
        return model

    def _generator_block(self, x, filters, upsample=True):
        """Bloc amélioré du générateur avec skip-connections"""
        # Shortcut pour la connexion résiduelle
        shortcut = x
        if upsample:
            shortcut = layers.UpSampling2D(size=(2, 2))(shortcut)
            shortcut = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        elif filters != x.shape[-1]:
            shortcut = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Chemin principal
        if upsample:
            x = layers.UpSampling2D(size=(2, 2))(x)
        
        # Première convolution
        x = layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Seconde convolution
        x = layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Ajout de la connexion résiduelle
        x = layers.add([x, shortcut])
        x = layers.LeakyReLU(0.2)(x)
        
        # Dropout pour régularisation (modéré pour éviter le sous-apprentissage)
        x = layers.Dropout(0.1)(x)
        
        return x
        
    def build_discriminator(self):
        """Discriminateur avec normalisation spectrale pour stabilité"""
        img_input = layers.Input(shape=self.input_shape)
        
        # Fonction auxiliaire pour créer les blocs de convolution avec normalisation spectrale
        def conv_block(x, filters, kernel_size=4, strides=2):
            if self.use_spectral_norm:
                x = SpectralNormalization(layers.Conv2D(filters, kernel_size, strides=strides, 
                                                     padding='same', use_bias=False))(x)
            else:
                x = layers.Conv2D(filters, kernel_size, strides=strides, 
                                 padding='same', use_bias=False)(x)
            
            x = layers.LeakyReLU(0.2)(x)
            return x
        
        # Architecture progressive
        x = conv_block(img_input, 64, strides=2)    # H/2
        x = conv_block(x, 128, strides=2)           # H/4
        x = conv_block(x, 256, strides=2)           # H/8
        x = conv_block(x, 512, strides=1)           # Maintient la taille pour plus de stabilité
        
        # Classification finale
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)  # Dropout pour régulariser
        
        # Type de sortie selon le mode (Wasserstein ou GAN classique)
        if self.use_wasserstein:
            output = layers.Dense(1, activation=None)(x)
        else:
            output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(img_input, output, name="Discriminator_Enhanced")
        
        # Compilation adaptée au mode
        if self.use_wasserstein:
            model.compile(
                loss=self._wasserstein_loss,
                optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
                metrics=['accuracy']
            )
        else:
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                metrics=['accuracy']
            )
        
        return model
        
    def _wasserstein_loss(self, y_true, y_pred):
        """Perte Wasserstein"""
        return tf.reduce_mean(y_true * y_pred)
    
    def build_combined_model(self):
        """Construction du modèle combiné pour l'entraînement du générateur"""
        self.discriminator.trainable = False
        
        # Créer le flux: générateur -> discriminateur
        z_input = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z_input)
        validity = self.discriminator(img)
        
        combined = models.Model(z_input, validity)
        
        # Compilation adaptée au mode
        if self.use_wasserstein:
            combined.compile(
                loss=self._wasserstein_loss,
                optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            )
        else:
            combined.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            )
        
        return combined
    
    def preprocess_batch_data(self, batch_data):
        """
        Prétraite les données de batch pour le modèle GAN
        Convertit de PyTorch à TensorFlow si nécessaire
        """
        try:
            if isinstance(batch_data, (list, tuple)):
                imgs = batch_data[0]  # Pour les loaders qui retournent (images, _)
            else:
                imgs = batch_data
            
            # Vérifier si c'est un tensor PyTorch et convertir si nécessaire
            if isinstance(imgs, torch.Tensor):
                # Convertir PyTorch tensor en numpy array
                imgs_np = imgs.cpu().detach().numpy()
                
                # Si les images sont au format NCHW (PyTorch), les convertir en NHWC (TensorFlow)
                if len(imgs_np.shape) == 4 and imgs_np.shape[1] in [1, 3]:
                    # Permuter les dimensions de NCHW à NHWC
                    imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
                    
                # Convertir numpy en TensorFlow tensor
                imgs = tf.convert_to_tensor(imgs_np, dtype=tf.float32)
            
            # Normaliser si nécessaire dans [-1, 1]
            min_val, max_val = tf.reduce_min(imgs), tf.reduce_max(imgs)
            if min_val < -1.0 or max_val > 1.0:
                logger.warning(f"Valeurs hors plage [-1, 1]: [{min_val}, {max_val}], normalisation appliquée")
                imgs = 2.0 * (imgs - min_val) / (max_val - min_val) - 1.0
            elif min_val >= 0.0 and max_val <= 1.0:
                # Conversion de [0, 1] à [-1, 1]
                logger.info(f"Valeurs normalisées de [{min_val}, {max_val}] à (-1, 1)")
                imgs = 2.0 * imgs - 1.0
            
            return imgs
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {e}")
            raise
    
    def train(self, train_loader, epochs=20, log_interval=50, plot_interval=500, save_interval=5):
        """
        Entraînement du GAN avec techniques de stabilité améliorées
        
        Args:
            train_loader: DataLoader contenant les images d'entraînement
            epochs: Nombre d'époques d'entraînement
            log_interval: Intervalle d'affichage des logs (en batches)
            plot_interval: Intervalle de génération d'images (en batches)
            save_interval: Intervalle de sauvegarde du modèle (en époques)
        
        Returns:
            metrics_history: Historique des métriques d'entraînement
        """
        # Vérifier le dataloader
        if train_loader is None or len(train_loader) == 0:
            logger.warning("Dataloader vide, entraînement impossible")
            return self.metrics_history
        
        # Créer deux optimiseurs distincts avec des taux d'apprentissage différents
        discriminator_lr = 0.00005 if self.use_wasserstein else 0.0001  # Taux réduit pour le discriminateur
        generator_lr = 0.0002  # Taux standard pour le générateur
        
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr, beta_1=0.5, beta_2=0.999)
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr, beta_1=0.5, beta_2=0.999)
        
        # Répertoire pour les checkpoints et samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", f"cnn_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        sample_dir = os.path.join("samples", f"cnn_{timestamp}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Paramètre pour le gradient penalty (uniquement pour Wasserstein)
        gradient_penalty_weight = 10.0
        
        # Set labels based on GAN type
        if self.use_wasserstein:
            # For WGAN: real=1, fake=-1
            real_labels = 1.0
            fake_labels = -1.0
        else:
            # For standard GAN: real=1, fake=0
            real_labels = 1.0
            fake_labels = 0.0
        
        # Fonction pour ajouter du bruit aux données (aide à la stabilité)
        def add_noise(images, stddev=0.05):
            noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=stddev)
            return images + noise
        
        # Fonction pour calculer le gradient penalty (Wasserstein)
        def gradient_penalty(real_images, fake_images):
            batch_size = tf.shape(real_images)[0]
            
            # Interpolation entre images réelles et fausses
            alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
            interpolated = real_images + alpha * (fake_images - real_images)
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.discriminator(interpolated, training=True)
                
            # Calcul du gradient et de sa norme
            grads = gp_tape.gradient(pred, interpolated)
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
            
            # Pénalité = (|grad| - 1)^2
            gp = tf.reduce_mean(tf.square(norm - 1.0))
            return gp
        
        # Variables pour suivi de la progression
        batch_count = 0
        total_batches = epochs * len(train_loader)
        
        # Boucle d'entraînement principale
        for epoch in range(epochs):
            start_time = datetime.now()
            d_losses_epoch = []
            g_losses_epoch = []
            d_accs_epoch = []
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Incrémenter le compteur global de batches
                batch_count += 1
                
                # Extraire les vraies images et ajouter du bruit
                real_images = self.preprocess_batch_data(batch_data)
                batch_size = tf.shape(real_images)[0]
                real_images_noisy = add_noise(real_images)
                
                # ====================
                #  Entraîner le discriminateur (moins fréquemment)
                # ====================
                # Réduit la fréquence d'entraînement du discriminateur si déséquilibre détecté
                train_discriminator = True
                if batch_idx > 20 and tf.reduce_mean(d_losses_epoch[-10:]) < -15.0:
                    # Discriminateur trop fort, le ralentir
                    if batch_idx % 3 != 0:  # Entraîner seulement 1 fois sur 3
                        train_discriminator = False
                
                if train_discriminator:
                    # Générer du bruit et créer des fausses images
                    noise = tf.random.normal([batch_size, self.latent_dim])
                    fake_images = self.generator(noise, training=True)
                    # Ajouter du bruit aux fausses images pour la stabilité
                    fake_images_noisy = add_noise(fake_images)
                    
                    # Entraîner le discriminateur avec GradientTape
                    with tf.GradientTape() as disc_tape:
                        if self.use_wasserstein:
                            # Pour Wasserstein GAN
                            real_preds = self.discriminator(real_images_noisy, training=True)
                            fake_preds = self.discriminator(fake_images_noisy, training=True)
                            
                            # Wasserstein loss: maximiser D(real) - D(fake)
                            d_loss = -tf.reduce_mean(real_preds) + tf.reduce_mean(fake_preds)
                            
                            # Ajouter le gradient penalty
                            gp = gradient_penalty(real_images, fake_images)
                            d_loss += gradient_penalty_weight * gp
                        else:
                            # Pour GAN standard
                            real_preds = self.discriminator(real_images_noisy, training=True)
                            fake_preds = self.discriminator(fake_images_noisy, training=True)
                            
                            # Binary cross-entropy loss
                            real_loss = tf.keras.losses.binary_crossentropy(
                                tf.ones_like(real_preds), real_preds
                            )
                            fake_loss = tf.keras.losses.binary_crossentropy(
                                tf.zeros_like(fake_preds), fake_preds
                            )
                            
                            d_loss = tf.reduce_mean(real_loss + fake_loss)
                    
                    # Appliquer les gradients au discriminateur
                    d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
                    disc_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
                    
                    # Calculer précision (uniquement pour GAN standard)
                    if not self.use_wasserstein:
                        real_acc = tf.reduce_mean(tf.cast(real_preds > 0.5, tf.float32))
                        fake_acc = tf.reduce_mean(tf.cast(fake_preds < 0.5, tf.float32))
                        d_acc = 0.5 * (real_acc + fake_acc)
                    else:
                        d_acc = 0.5  # Placeholder pour WGAN
                
                # ====================
                #  Entraîner le générateur
                # ====================
                # Générer du nouveau bruit
                noise = tf.random.normal([batch_size, self.latent_dim])
                
                # Entraîner le générateur avec GradientTape
                with tf.GradientTape() as gen_tape:
                    # Générer des fausses images
                    gen_images = self.generator(noise, training=True)
                    
                    # Obtenir les prédictions du discriminateur
                    gen_preds = self.discriminator(gen_images, training=False)
                    
                    # Calcul de la perte du générateur
                    if self.use_wasserstein:
                        # Pour Wasserstein: maximiser les sorties du discriminateur
                        g_loss = -tf.reduce_mean(gen_preds)
                    else:
                        # Pour GAN standard: tromper le discriminateur
                        g_loss = tf.reduce_mean(
                            tf.keras.losses.binary_crossentropy(
                                tf.ones_like(gen_preds), gen_preds
                            )
                        )
                
                # Appliquer les gradients au générateur
                g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
                
                # Logging et sauvegarde
                d_losses_epoch.append(float(d_loss))
                g_losses_epoch.append(float(g_loss))
                if not self.use_wasserstein:
                    d_accs_epoch.append(float(d_acc))
                
                # Afficher les statistiques périodiquement
                if batch_idx % log_interval == 0:
                    # Déterminer l'équilibre GAN
                    d_g_ratio = abs(float(d_loss) / float(g_loss)) if float(g_loss) != 0 else float('inf')
                    balance = "équilibré" if 0.5 <= d_g_ratio <= 2.0 else "déséquilibré"
                    
                    logger.warning(
                        f"[Epoch {epoch+1}/{epochs}] "
                        f"[Batch {batch_idx}/{len(train_loader)}] "
                        f"[D loss: {float(d_loss):.4f}] "
                        f"[G loss: {float(g_loss):.4f}] "
                        f"[Balance GAN: {balance}]"
                    )
                    
                    # Sauvegarder pour le traçage
                    self.metrics_history['d_loss'].append(float(d_loss))
                    self.metrics_history['g_loss'].append(float(g_loss))
                    if not self.use_wasserstein:
                        self.metrics_history['d_acc'].append(float(d_acc))
                
                # Générer et sauvegarder des échantillons périodiquement
                if batch_count % plot_interval == 0:
                    self.save_sample_images(epoch)
            
            # Fin de l'époque: calculer les moyennes
            epoch_d_loss = np.mean(d_losses_epoch)
            epoch_g_loss = np.mean(g_losses_epoch)
            epoch_d_acc = np.mean(d_accs_epoch) if d_accs_epoch else 0.0
            
            # Sauvegarder les moyennes d'époque
            self.metrics_history['epoch_d_loss'].append(epoch_d_loss)
            self.metrics_history['epoch_g_loss'].append(epoch_g_loss)
            if not self.use_wasserstein:
                self.metrics_history['epoch_d_acc'].append(epoch_d_acc)
            
            # Temps écoulé pour cette époque
            time_taken = datetime.now() - start_time
            logger.info(
                f"[Époque {epoch+1}/{epochs}] "
                f"[D loss: {epoch_d_loss:.4f}] "
                f"[G loss: {epoch_g_loss:.4f}] "
                f"[Temps: {time_taken}]"
            )
            
            # Sauvegarder le modèle périodiquement
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}'))
        
        # Tracer l'historique d'entraînement
        self.plot_training_history()
        
        return self.metrics_history

    def generate_images(self, num_images=16, noise=None, truncation=0.8):
        """
        Génère des images à partir du générateur
        
        Args:
            num_images: Nombre d'images à générer
            noise: Vecteur de bruit (optionnel, sera généré si None)
            truncation: Facteur de troncation pour la génération
        
        Returns:
            Images générées sous forme de numpy array
        """
        try:
            if noise is None:
                # Générer du bruit avec troncation pour améliorer la qualité
                noise = tf.random.truncated_normal([num_images, self.latent_dim], stddev=truncation)
            elif isinstance(noise, np.ndarray):
                # Convertir en tensor TF si nécessaire
                noise = tf.convert_to_tensor(noise, dtype=tf.float32)
            
            # Générer les images
            generated_images = self.generator(noise, training=False)
            
            # Convertir en numpy array
            images = generated_images.numpy()
            
            # Normaliser dans [0, 1] pour l'affichage si nécessaire
            if np.min(images) < 0:
                images = (images + 1) / 2.0
            
            return images
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'images: {e}")
            return None

    def save_images(self, images, filepath):
        """
        Sauvegarde un ensemble d'images dans un fichier
        
        Args:
            images: Array d'images à sauvegarder
            filepath: Chemin de sauvegarde
        """
        try:
            # S'assurer que le répertoire existe
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Normaliser si nécessaire
            if np.min(images) < 0 or np.max(images) > 1:
                images = (images + 1) / 2.0
                
            # Supposons que les images sont dans le format (N, H, W, C)
            # et limitons à 16 images maximum
            num_images = min(16, images.shape[0])
            r = int(np.sqrt(num_images))
            c = int(np.ceil(num_images / r))
            
            fig, axs = plt.subplots(r, c, figsize=(c * 2, r * 2))
            cnt = 0
            
            for i in range(r):
                for j in range(c):
                    if cnt < num_images:
                        if r == 1:
                            axs[j].imshow(images[cnt])
                            axs[j].axis('off')
                        else:
                            axs[i, j].imshow(images[cnt])
                            axs[i, j].axis('off')
                        cnt += 1
            
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des images: {e}")

    def save_sample_images(self, epoch):
        """
        Génère et sauvegarde des échantillons d'images
        
        Args:
            epoch: Numéro de l'époque actuelle
        """
        r, c = 4, 4  # Grille de 4x4
        noise = tf.random.normal([r * c, self.latent_dim])
        gen_imgs = self.generate_images(r * c, noise)
        
        # Créer le répertoire de sauvegarde
        sample_dir = os.path.join("samples", f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Sauvegarder les images
        filename = os.path.join(sample_dir, f'epoch_{epoch+1}.png')
        self.save_images(gen_imgs, filename)

    def save_model(self, filepath):
        """
        Sauvegarde le modèle GAN
        
        Args:
            filepath: Chemin de sauvegarde de base (sans extension)
        """
        try:
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Sauvegarder le générateur et le discriminateur
            self.generator.save(f"{filepath}_generator.keras")
            self.discriminator.save(f"{filepath}_discriminator.keras")
            
            # Sauvegarder les hyperparamètres
            config = {
                'input_shape': self.input_shape,
                'latent_dim': self.latent_dim,
                'use_spectral_norm': self.use_spectral_norm,
                'use_wasserstein': self.use_wasserstein,
                'metrics_history': self.metrics_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{filepath}_config.json", 'w') as f:
                import json
                json.dump(config, f)
                
            logger.info(f"Modèle sauvegardé dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")

    def load_model(self, filepath):
        """
        Charge un modèle GAN précédemment sauvegardé
        
        Args:
            filepath: Chemin de chargement de base (sans extension)
        """
        try:
            # Charger le générateur et le discriminateur
            self.generator = tf.keras.models.load_model(f"{filepath}_generator.keras")
            self.discriminator = tf.keras.models.load_model(f"{filepath}_discriminator.keras")
            
            # Reconstruire le modèle combiné
            self.combined = self.build_combined_model()
            
            # Charger les hyperparamètres
            with open(f"{filepath}_config.json", 'r') as f:
                import json
                config = json.load(f)
                
            self.input_shape = config['input_shape']
            self.latent_dim = config['latent_dim']
            self.use_spectral_norm = config['use_spectral_norm']
            self.use_wasserstein = config['use_wasserstein']
            self.metrics_history = config['metrics_history']
            
            logger.info(f"Modèle chargé depuis {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False

    def plot_training_history(self):
        """Trace les courbes d'apprentissage"""
        try:
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Tracer les pertes
            if self.metrics_history['d_loss'] and self.metrics_history['g_loss']:
                ax1.plot(self.metrics_history['d_loss'], label='D loss', alpha=0.5)
                ax1.plot(self.metrics_history['g_loss'], label='G loss', alpha=0.5)
                
                # Ajouter moyennes mobiles
                import pandas as pd
                d_loss_smooth = pd.Series(self.metrics_history['d_loss']).rolling(window=50).mean()
                g_loss_smooth = pd.Series(self.metrics_history['g_loss']).rolling(window=50).mean()
                
                ax1.plot(d_loss_smooth, label='D loss (avg)', linewidth=2)
                ax1.plot(g_loss_smooth, label='G loss (avg)', linewidth=2)
                
                ax1.set_title('Pertes d\'entraînement')
                ax1.set_xlabel('Batch')
                ax1.set_ylabel('Perte')
                ax1.legend()
                ax1.grid(alpha=0.3)
            
            # Tracer la précision du discriminateur (si disponible)
            if not self.use_wasserstein and self.metrics_history['d_acc']:
                ax2.plot(self.metrics_history['d_acc'], label='D accuracy', alpha=0.5)
                
                # Ajouter moyenne mobile
                import pandas as pd
                d_acc_smooth = pd.Series(self.metrics_history['d_acc']).rolling(window=50).mean()
                ax2.plot(d_acc_smooth, label='D accuracy (avg)', linewidth=2)
                
                ax2.set_title('Précision du discriminateur')
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Précision')
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Sauvegarder la figure
            save_dir = "logs"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=150)
            plt.close()
        
        except Exception as e:
            logger.error(f"Erreur lors du traçage de l'historique: {e}")