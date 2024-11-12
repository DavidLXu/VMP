from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from vmp.motion_loader import MotionDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Define Residual Block for deeper feature extraction
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, window_length=61):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.norm1 = nn.LayerNorm([out_channels, window_length])
        self.norm2 = nn.LayerNorm([out_channels, window_length])
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

# Encoder
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, window_length):
        super(VAE_Encoder, self).__init__()
        # Initial conv layer without bias
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 4 ConvResNet blocks
        self.res_block1 = ResBlock(64, 64, window_length=window_length)
        self.res_block2 = ResBlock(64, 64, window_length=window_length) 
        self.res_block3 = ResBlock(64, 64, window_length=window_length)
        self.res_block4 = ResBlock(64, 64, window_length=window_length)
        
        # Layer normalization
        self.norm = nn.LayerNorm(64)
        
        # Add global average pooling before final linear layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Modify final linear layers to take flattened input
        self.fc_mu = nn.Linear(64, latent_dim)  # Now takes pooled features
        self.fc_logvar = nn.Linear(64, latent_dim)
        
    def forward(self, x):
        # Initial conv + activation
        x = F.relu(self.conv1(x))
        
        # Pass through ResNet blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Layer norm and final activation
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        x = F.relu(x)
        
        # Global average pooling
        x = self.global_pool(x)  # Output shape: (batch_size, 64, 1)
        x = x.squeeze(-1)  # Output shape: (batch_size, 64)
        
        # Get single latent vector per sequence
        mu = self.fc_mu(x)  # Output shape: (batch_size, latent_dim)
        logvar = self.fc_logvar(x)  # Output shape: (batch_size, latent_dim)
        return mu, logvar

# Decoder
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, window_length=61):
        super(VAE_Decoder, self).__init__()
        self.window_length = window_length
        self.fc = nn.Linear(latent_dim, 64 * window_length)  # Expand latent to full sequence length
        
        # 4 ConvResNet blocks
        self.res_block1 = ResBlock(64, 64)
        self.res_block2 = ResBlock(64, 64)
        self.res_block3 = ResBlock(64, 64)
        self.res_block4 = ResBlock(64, 64)
        
        # Layer normalization
        self.norm = nn.LayerNorm(64)
        
        # Final conv without bias
        self.conv = nn.Conv1d(64, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, z):
        # Reshape to sequence
        x = F.relu(self.fc(z))  # Output: (batch_size, 64 * window_length)
        x = x.view(x.size(0), 64, self.window_length)  # Reshape to (batch_size, 64, window_length)
        
        # Pass through ResNet blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Layer norm and activation
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        x = F.relu(x)
        
        # Final conv
        x = self.conv(x)
        return x

# VAE Class combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, window_length=61):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim, latent_dim, window_length)
        self.decoder = VAE_Decoder(latent_dim, input_dim, window_length)  # Pass window_length to decoder
        self.beta = 0.002  # KL weight for beta-VAE

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Reshape input from (batch_size, window_length, feature_dim) to (batch_size, feature_dim, window_length)
        x = x.transpose(1, 2)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        # Reshape output back to (batch_size, window_length, feature_dim)
        x_reconstructed = x_reconstructed.transpose(1, 2)
        return x_reconstructed, mu, logvar

# Loss function for beta-VAE with KL weight
def vae_loss(reconstructed, original, mu, logvar, beta=0.002):
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence, recon_loss, kl_divergence

def save_encoder_state(vae, window_length=61, path='encoder_only.pt'):
    encoder_state = {
        'encoder_state_dict': vae.encoder.state_dict(),
        'input_dim': vae.encoder.conv1.in_channels,
        'latent_dim': vae.encoder.fc_mu.out_features,
        'window_length': window_length
    }
    torch.save(encoder_state, path)
    print(f"Encoder state saved to {path}")

if __name__ == "__main__":
    # Training setup with RAdam optimizer and cosine annealing
    motion_combo = "/data/ASE/ase/data/motions/walk/dataset_reallusion_walk.yaml"
    motion_clip = "/data/ASE/ase/data/motions/walk/RL_Avatar_WalkForward01_Motion.npy"

    W = 30
    window_length = 2 * W + 1
    input_dim = 81 # 93
    motion_dataset = MotionDataset(motion_file=motion_combo,
                            num_motions=1000,
                            window_length=window_length)  # 1 second window

    motion_dataloader = DataLoader(motion_dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE(input_dim=input_dim, window_length=window_length).to(device)  # input_dim is feature_dim since we transpose

    # Print total number of parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total number of parameters: {total_params:,}")

    optimizer = torch.optim.RAdam(vae.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # Initialize tensorboard writer
    writer = SummaryWriter('runs/vae_training')

    num_epochs = 200
    best_loss = float('inf')

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        for batch in motion_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            reconstructed, mu, logvar = vae(batch)
            print(batch.shape, mu.shape)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar, beta=vae.beta)
            
            if torch.isnan(loss):
                print(f"NaN loss detected!")
                print(f"mu stats: min={mu.min()}, max={mu.max()}, mean={mu.mean()}")
                print(f"logvar stats: min={logvar.min()}, max={logvar.max()}, mean={logvar.mean()}")
                print(f"reconstructed stats: min={reconstructed.min()}, max={reconstructed.max()}, mean={reconstructed.mean()}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        # Log training metrics
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/reconstruction', avg_recon_loss, epoch)
        writer.add_scalar('Loss/kl_divergence', avg_kl_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, 'vae_best.pt')
            # Additionally save encoder state separately
            save_encoder_state(vae, window_length=window_length, path='encoder_best.pt')
            print(f"Checkpoint saved at epoch {epoch + 1} with loss {best_loss:.4f}")

        # Evaluation mode
        if (epoch + 1) % 10 == 0:  # Evaluate every 10 epochs
            vae.eval()
            eval_loss = 0
            eval_recon_loss = 0
            eval_kl_loss = 0
            num_eval = 0
            with torch.no_grad():
                for batch in motion_dataloader:  # Using same dataloader for simplicity
                    batch = batch.to(device)
                    reconstructed, mu, logvar = vae(batch)
                    loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar, beta=vae.beta)
                    eval_loss += loss.item()
                    eval_recon_loss += recon_loss.item()
                    eval_kl_loss += kl_loss.item()
                    num_eval += 1
            
            avg_eval_loss = eval_loss / num_eval
            avg_eval_recon_loss = eval_recon_loss / num_eval
            avg_eval_kl_loss = eval_kl_loss / num_eval
            
            # Log evaluation metrics
            writer.add_scalar('Loss/eval', avg_eval_loss, epoch)
            writer.add_scalar('Loss/eval_reconstruction', avg_eval_recon_loss, epoch)
            writer.add_scalar('Loss/eval_kl_divergence', avg_eval_kl_loss, epoch)
            
            print(f"Evaluation Loss at epoch {epoch + 1}: {avg_eval_loss:.4f}")

    writer.close()
