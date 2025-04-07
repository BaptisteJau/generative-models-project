import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import torch

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
        
    def build_generator(self):
        """Build the generator network that transforms noise into images"""
        noise_shape = (self.latent_dim,)
        
        model = models.Sequential(name="Generator")
        
        # First dense layer
        model.add(layers.Dense(4 * 4 * 256, input_shape=noise_shape))
        model.add(layers.Reshape((4, 4, 256)))
        
        # Upsampling convolutional layers
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        # Output layer with tanh activation for pixel values in [-1, 1]
        model.add(layers.Conv2D(self.input_shape[2], (3, 3), padding='same', activation='tanh'))
        
        return model
        
    def build_discriminator(self):
        """Build the discriminator network to classify real/fake images"""
        model = models.Sequential(name="Discriminator")
        
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', 
                             input_shape=self.input_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        
        # Output layer for binary classification (real/fake)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile discriminator
        model.compile(loss='binary_crossentropy',
                     optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     metrics=['accuracy'])
                     
        return model
        
    def build_combined_model(self):
        """Build combined model for training the generator"""
        # For the combined model, we only train the generator
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
    
    def train(self, train_loader, epochs=100, log_interval=10, plot_interval=100):
        """Train the GAN model"""
        # Array pour stocker l'historique de l'entraînement
        history = {"d_loss": [], "g_loss": []}
        
        batch_size = train_loader.batch_size
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                # Gérer les deux types de retour possibles du dataloader
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    imgs, _ = batch  # Loader retourne (images, labels)
                else:
                    imgs = batch  # Loader retourne uniquement les images
                
                # Conversion des données PyTorch au format Keras
                if isinstance(imgs, torch.Tensor):
                    # Convertir PyTorch tensor -> Numpy
                    imgs = imgs.detach().cpu().numpy()
                    
                    # Vérifier et modifier l'ordre des dimensions si nécessaire
                    if imgs.shape[1] in [1, 3]:  # Format NCHW (channels en 2e position)
                        imgs = np.transpose(imgs, (0, 2, 3, 1))
                        print(f"Images converties de NCHW -> NHWC: {imgs.shape}")
                    
                    # Normaliser si nécessaire pour tanh (-1 à 1)
                    if imgs.min() >= 0 and imgs.max() <= 1:
                        imgs = imgs * 2 - 1
                
                # Utiliser directement ces images converties avec TensorFlow
                imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
                print(f"Forme finale des images: {imgs.shape}")
                
                # Générer du bruit aléatoire
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # Générer des images fausses
                gen_imgs = self.generator.predict(noise, verbose=0)
                
                # Entraîner le discriminateur
                d_loss_real = self.discriminator.train_on_batch(imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Entraîner le générateur
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, real)
                
                # Enregistrer les pertes
                history["d_loss"].append(d_loss[0])
                history["g_loss"].append(g_loss)
                
                # Afficher les progrès
                if i % log_interval == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(train_loader)}] "
                          f"[D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] "
                          f"[G loss: {g_loss:.4f}]")
                
                # Générer et sauvegarder des images à certains intervalles
                if i % plot_interval == 0:
                    self.save_sample_images(epoch)
                    
        return history
    
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
        """Save the generator and discriminator models
        
        Args:
            filepath: Base filepath for saving models
        """
        self.generator.save(f"{filepath}_generator.h5")
        self.discriminator.save(f"{filepath}_discriminator.h5")
        
    def load_model(self, filepath):
        """Load generator and discriminator models
        
        Args:
            filepath: Base filepath for loading models
        """
        from tensorflow.keras.models import load_model
        self.generator = load_model(f"{filepath}_generator.h5")
        self.discriminator = load_model(f"{filepath}_discriminator.h5")
        
        # Rebuild the combined model
        self.combined = self.build_combined_model()