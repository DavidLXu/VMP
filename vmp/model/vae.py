import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Residual Block for deeper feature extraction
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

# Encoder
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResBlock(64, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res_block2 = ResBlock(128, 128)
        
        # Linear layers to get mean and log-variance
        self.fc_mu = nn.Linear(128 * (input_dim // 4), latent_dim)  # Assumes input_dim is divisible by 4
        self.fc_logvar = nn.Linear(128 * (input_dim // 4), latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = F.relu(self.conv2(x))
        x = self.res_block2(x)
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * (output_dim // 4))
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res_block1 = ResBlock(64, 64)
        self.deconv2 = nn.ConvTranspose1d(64, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 128, -1)  # Reshape to fit ConvTranspose
        x = F.relu(self.deconv1(x))
        x = self.res_block1(x)
        x = self.deconv2(x)
        return x

# VAE Class combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Loss function for VAE (reconstruction + KL divergence)
def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence



# Example training loop
vae = VAE(input_dim=your_input_dim, latent_dim=64).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for batch in data_loader:  # Assumes data_loader provides batches of motion sequences
        batch = batch.to(device)
        optimizer.zero_grad()
        
        reconstructed, mu, logvar = vae(batch)
        loss = vae_loss(reconstructed, batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
