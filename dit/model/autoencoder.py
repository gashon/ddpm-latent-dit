import torch
from einops import rearrange
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()

        self.linear = nn.Linear(latent_dim, 32 * 7 * 7)

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 1 x 28 x 28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        x = self.linear(z)
        x = rearrange(x, "b (c h w) -> b c h w", c=16, h=7, w=7)
        return self.upconv(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, image_size: int = 28):
        super().__init__()

        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
