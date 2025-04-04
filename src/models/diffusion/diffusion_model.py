import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class UNet(nn.Module):
    """U-Net architecture for the noise prediction network in diffusion models"""
    
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.downs = nn.ModuleList([
            DownBlock(64, 128, time_dim),
            DownBlock(128, 256, time_dim),
            DownBlock(256, 512, time_dim),
            DownBlock(512, 512, time_dim),
        ])
        
        # Bottleneck
        self.bottleneck = BottleneckBlock(512, 512, time_dim)
        
        # Decoder (upsampling)
        self.ups = nn.ModuleList([
            UpBlock(512 + 512, 512, time_dim),
            UpBlock(512 + 256, 256, time_dim),
            UpBlock(256 + 128, 128, time_dim),
            UpBlock(128 + 64, 64, time_dim),
        ])
        
        # Final convolution
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        h = self.conv_in(x)
        skips = [h]
        
        # Encoder
        for down in self.downs:
            h = down(h, t_emb)
            skips.append(h)
            
        # Bottleneck
        h = self.bottleneck(h, t_emb)
        
        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb)
            
        # Final convolution
        return self.conv_out(h)

class DownBlock(nn.Module):
    """Downsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return self.pool(h)

class BottleneckBlock(nn.Module):
    """Bottleneck block for U-Net"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
    def forward(self, x, t_emb):
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return h

class UpBlock(nn.Module):
    """Upsampling block for U-Net"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.act1(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(self.conv2(h)))
        return h

class DiffusionModel:
    def __init__(self, config=None):
        """Initialize a diffusion model with specified configuration"""
        # Default config
        self.config = {
            "image_size": 64,
            "num_channels": 3,
            "batch_size": 32,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "num_timesteps": 1000,
            "learning_rate": 1e-4,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Update with provided config
        if config is not None:
            self.config.update(config)
            
        # Initialize the noise prediction model (U-Net)
        self.model = self.build_model()
        self.model.to(self.config["device"])
        
        # Setup the optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        # Setup the noise schedule (betas)
        self.betas = self._get_noise_schedule()
        
        # Precompute values for inference
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def build_model(self):
        """Build the noise prediction U-Net model"""
        return UNet(
            in_channels=self.config.get("num_channels", 3),
            out_channels=self.config.get("num_channels", 3), 
            time_dim=256
        )
    
    def _get_noise_schedule(self):
        """Get the noise schedule for the diffusion process"""
        return torch.linspace(
            self.config["beta_start"],
            self.config["beta_end"],
            self.config["num_timesteps"]
        )
    
    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process: add noise to images according to timestep
        
        Args:
            x_0: clean images [B, C, H, W]
            t: timesteps [B]
            
        Returns:
            x_t: noised images [B, C, H, W]
            noise: noise added to images [B, C, H, W]
        """
        # Get batch size
        batch_size = x_0.shape[0]
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Extract coefficients for this timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Forward process: add noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def train(self, data_loader, epochs=100):
        """
        Train the diffusion model
        
        Args:
            data_loader: PyTorch DataLoader for training data
            epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        self.model.train()
        device = self.config["device"]
        history = {"loss": []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for i, batch in enumerate(progress_bar):
                # Get batch of images and move to device
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(device)
                else:
                    images = batch.to(device)
                    
                # Ensure images are in the range [-1, 1]
                if images.min() >= 0 and images.max() <= 1:
                    images = 2 * images - 1
                
                # Sample random timesteps
                t = torch.randint(0, self.config["num_timesteps"], (images.shape[0],), device=device)
                
                # Add noise to images
                x_t, noise = self.forward_diffusion(images, t)
                
                # Predict noise using model
                predicted_noise = self.model(x_t, t)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (i + 1)})
                
            # Record average loss for this epoch
            avg_loss = epoch_loss / len(data_loader)
            history["loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Save sample generations every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_samples(n=4, save_path=f"samples/diffusion_epoch_{epoch+1}.png")
                
        return history
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Sample from p(x_{t-1} | x_t) using the model
        
        Args:
            x: Current noisy images [B, C, H, W]
            t: Current timesteps [B]
            t_index: Timestep index (integer)
            
        Returns:
            Sample from p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Predict the noise
        predicted_noise = self.model(x, t)
        
        # Compute the mean for p(x_{t-1} | x_t)
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Add noise only if t > 0
        if t_index > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def generate_samples(self, n=1, save_path=None):
        """
        Generate samples from the model
        
        Args:
            n: Number of samples to generate
            save_path: Path to save the generated images (optional)
            
        Returns:
            Generated images tensor [N, C, H, W] with values in [-1, 1]
        """
        self.model.eval()
        device = self.config["device"]
        img_size = self.config["image_size"]
        channels = self.config["num_channels"]
        
        # Start from pure noise
        x = torch.randn(n, channels, img_size, img_size, device=device)
        
        # Gradually denoise the images
        for i in tqdm(reversed(range(0, self.config["num_timesteps"])), desc='Sampling'):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
            
        # Rescale to [0, 1] for visualization
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            from torchvision.utils import save_image
            save_image(x, save_path)
            
        return x
    
    def save_model(self, filepath):
        """Save the model and optimizer state"""
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(state_dict, filepath)
        
    def load_model(self, filepath):
        """Load the model and optimizer state"""
        state_dict = torch.load(filepath, map_location=self.config["device"])
        
        # Update config
        self.config.update(state_dict['config'])
        
        # Rebuild model with new config if needed
        self.model = self.build_model()
        self.model.to(self.config["device"])
        
        # Load state dictionaries
        self.model.load_state_dict(state_dict['model'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        # Rebuild noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)