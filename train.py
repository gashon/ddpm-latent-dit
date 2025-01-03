import argparse
import logging
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dit.config import Config
from dit.data.dataset import get_mnist_dataset
from dit.model.autoencoder import Autoencoder
from dit.model.diffusion import LatentDiffusion
from dit.model.transformer import DiffusionTransformer
from dit.utils.training import AverageMeter, save_checkpoint


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


def train_autoencoder(epochs: int, batch_size: int, lr: float):
    device = Config.DEVICE
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LR = lr

    train_dataset, _ = get_mnist_dataset(train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    autoencoder = Autoencoder(
        image_size=Config.IMAGE_SIZE, latent_dim=Config.LATENT_DIM
    ).to(device)

    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    recon_criterion = torch.nn.MSELoss()

    autoencoder.train()
    logger.info("Starting autoencoder-only training...")

    for epoch in range(epochs):
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"[AE] Epoch {epoch+1}/{epochs}", leave=False)
        for images, _ in pbar:
            images = images.to(device)
            recon = autoencoder(images)
            loss = recon_criterion(recon, images)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({"recon_loss": f"{loss_meter.avg:.4f}"})

        logger.info(
            f"[AE-Only] Epoch [{epoch+1}/{epochs}] - Recon Loss: {loss_meter.avg:.4f}"
        )

        save_path = os.path.join("checkpoints", f"ae_only_epoch_{epoch+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "autoencoder_state": autoencoder.state_dict(),
                "ae_optimizer_state": ae_optimizer.state_dict(),
            },
            save_path,
        )

    logger.info("Autoencoder-only training completed.")


def train_model(epochs: int, batch_size: int, lr: float):
    device = Config.DEVICE
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.LR = lr

    train_dataset, _ = get_mnist_dataset(train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    autoencoder = Autoencoder(
        image_size=Config.IMAGE_SIZE, latent_dim=Config.LATENT_DIM
    ).to(device)

    diffusion_transformer = DiffusionTransformer(
        latent_dim=Config.LATENT_DIM,
        embed_dim=Config.TRANSFORMER_EMBED_DIM,
        num_heads=Config.TRANSFORMER_NUM_HEADS,
        hidden_dim=Config.TRANSFORMER_HIDDEN_DIM,
        num_layers=Config.TRANSFORMER_NUM_LAYERS,
    ).to(device)

    diffusion_model = LatentDiffusion(
        transformer=diffusion_transformer,
        timesteps=Config.TIMESTEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
    ).to(device)

    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

    autoencoder.train()
    diffusion_model.train()
    logger.info("Starting joint autoencoder + diffusion model training...")

    for epoch in range(epochs):
        loss_meter = AverageMeter()
        pbar = tqdm(
            train_loader,
            desc=f"[Joint Diffusion] Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for images, _ in pbar:
            images = images.to(device)
            latents = autoencoder.encode(images)
            loss = diffusion_model.training_step(latents)

            diffusion_optimizer.zero_grad()
            ae_optimizer.zero_grad()
            loss.backward()
            diffusion_optimizer.step()
            ae_optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        logger.info(
            f"[Joint Diffusion] Epoch [{epoch+1}/{epochs}] - Diffusion Loss: {loss_meter.avg:.4f}"
        )

        save_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "autoencoder_state": autoencoder.state_dict(),
                "diffusion_state": diffusion_model.state_dict(),
                "ae_optimizer_state": ae_optimizer.state_dict(),
                "diffusion_optimizer_state": diffusion_optimizer.state_dict(),
            },
            save_path,
        )

    logger.info("Joint diffusion training completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Diffusion Transformer on MNIST"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--train_autoencoder_only",
        action="store_true",
        help="If set, only trains the autoencoder for reconstruction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_autoencoder_only:
        train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
