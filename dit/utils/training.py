import os

import matplotlib.pyplot as plt
import torch

from dit.config import Config
from dit.model.autoencoder import Autoencoder
from dit.model.diffusion import LatentDiffusion
from dit.model.transformer import DiffusionTransformer


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


@torch.no_grad()
def sample_images(ckpt_path, num_samples=16):
    checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)

    autoencoder = Autoencoder(
        image_size=Config.IMAGE_SIZE, latent_dim=Config.LATENT_DIM
    ).to(Config.DEVICE)
    diffusion_transformer = DiffusionTransformer(
        latent_dim=Config.LATENT_DIM,
        embed_dim=Config.TRANSFORMER_EMBED_DIM,
        num_heads=Config.TRANSFORMER_NUM_HEADS,
        hidden_dim=Config.TRANSFORMER_HIDDEN_DIM,
        num_layers=Config.TRANSFORMER_NUM_LAYERS,
    ).to(Config.DEVICE)
    diffusion_model = LatentDiffusion(
        transformer=diffusion_transformer,
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
    ).to(Config.DEVICE)

    # Load state
    autoencoder.load_state_dict(checkpoint["autoencoder_state"])
    diffusion_model.load_state_dict(checkpoint["diffusion_state"])

    # Generate latents from scratch
    latents = diffusion_model.p_sample_loop(
        shape=(num_samples, Config.LATENT_DIM), device=Config.DEVICE
    )
    # Decode to images
    images = autoencoder.decode(latents)

    # Plot a grid of generated images
    images = images.cpu().detach()
    os.makedirs("samples", exist_ok=True)
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(int(num_samples**0.5), int(num_samples**0.5), i + 1)
        plt.imshow(images[i][0], cmap="gray")
        plt.axis("off")
    sample_file = "samples/generated_samples.png"
    plt.tight_layout()
    plt.savefig(sample_file)
    print(f"Samples saved to {sample_file}")
    plt.close()
