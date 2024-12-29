import torch


class Config:
    DATASET_NAME = "MNIST"
    IMAGE_SIZE = 28
    LATENT_DIM = 64

    TIMESTEPS = 1_000
    BETA_START = 0.0001
    BETA_END = 0.02

    TRANSFORMER_EMBED_DIM = 64
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_HIDDEN_DIM = 128
    TRANSFORMER_NUM_LAYERS = 4

    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
