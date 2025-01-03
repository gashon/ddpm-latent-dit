import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiffusion(nn.Module):
    def __init__(
        self, transformer: nn.Module, timesteps: int, beta_start: float, beta_end: float
    ):
        super().__init__()
        self.transformer = transformer
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]]),
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffusion forward process: x_t = sqrt_alpha_cumprod * x_0 + sqrt(1 - alpha_cumprod) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def training_step(self, x_0):
        """
        1. sample random t
        2. sample noise
        3. generate x_t
        4. predict noise with transformer
        5. compute loss
        """

        b = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t)

        noise_pred = self.transformer(x_t, t.float())
        loss = F.mse_loss(noise, noise_pred)

        return loss

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple):
        out = a.gather(-1, t)
        return out.view(-1, *[1] * (len(x_shape) - 1))

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        Reverse process: x_{t-1} ~ N(mu_theta(x_t, t), sigma_t^2 I)
        """

        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        noise_pred = self.transformer(x_t, t.float())

        # x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t) * noise_pred / sqrt(1 - alpha_cumprod_t) ) + sqrt(beta_t) * z
        alpha_t = 1.0 - betas_t
        sqrt_alpha_t = torch.sqrt(alpha_t)

        mu = (1.0 / sqrt_alpha_t) * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred
        )

        sigma_t = torch.sqrt(betas_t)

        # Random noise, except if t=0
        z = torch.randn_like(x_t)

        # If t=0, no noise
        mask = (t > 0).float().unsqueeze(-1)
        x_prev = mu + mask * sigma_t * z

        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, device: str):
        b = shape[0]

        x = torch.randn_like(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full(i, (b,), device=device)
            x = self.p_sample(x, t)

        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.transformer(x, t)
