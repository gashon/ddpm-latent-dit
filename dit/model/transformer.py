import math

import torch
from einops import rearrange
from torch import nn


def rope_fn(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    b, s, d = x.shape
    assert d % 2 == 0, "RoPE requires the hidden dimension to be even."
    half_d = d // 2

    # Split the last dimension into two parts
    x1 = x[..., :half_d]  # [b, s, half_d]
    x2 = x[..., half_d:]  # [b, s, half_d]

    # Position indices: shape [s, 1]
    pos = torch.arange(s, device=x.device, dtype=x.dtype).unsqueeze(-1)

    # Compute rotation frequencies: shape [half_d/2]
    #   freq_seq[i] = 1 / (base^(2i / d))
    freq_seq = torch.arange(0, half_d, 2, device=x.device, dtype=x.dtype)
    freq_seq = 1.0 / (base ** (freq_seq / half_d))  # [half_d/2]

    # angles: [s, half_d/2]
    angles = pos * freq_seq.unsqueeze(0)

    # Precompute sin & cos
    sin = torch.sin(angles)  # [s, half_d/2]
    cos = torch.cos(angles)  # [s, half_d/2]

    # For proper broadcasting:
    #   we want [b, s, half_d] for each sin, cos
    #   but sin and cos currently are [s, half_d/2].
    # We next expand to [s, half_d/2, 1] and then broadcast over batch b:
    sin = rearrange(sin, "s m -> 1 s m 1")
    cos = rearrange(cos, "s m -> 1 s m 1")

    # Reshape x1 and x2 to align with the sin/cos dims
    # x1, x2: [b, s, half_d] -> [b, s, half_d/2, 1]
    half_d2 = half_d // 2
    x1 = x1.reshape(b, s, half_d2, 1)
    x2 = x2.reshape(b, s, half_d2, 1)

    # Apply rotary transformation
    #   x_rot_1 = x1 * cos - x2 * sin
    #   x_rot_2 = x2 * cos + x1 * sin
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin

    # Bring x1_rot, x2_rot back to [b, s, half_d]
    x1_rot = rearrange(x1_rot, "b s m 1 -> b s m")
    x2_rot = rearrange(x2_rot, "b s m 1 -> b s m")

    # Concatenate along the last dimension -> [b, s, d]
    return torch.cat([x1_rot, x2_rot], dim=-1)


class GQA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads."
        assert (
            hidden_dim % num_groups == 0
        ), "hidden_dim must be divisible by num_groups."
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups."

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.qdim = hidden_dim // num_heads  # = d_q
        self.kvdim = hidden_dim // num_groups  # = d_kv

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,  # [b, s, hidden_dim]
        attn_mask: torch.Tensor = None,  # [b, s] or [b, 1, s, s]
    ):

        b, s, _ = x.shape

        q = self.q_proj(x)  # [b, s, d_model]
        k = self.k_proj(x)  # [b, s, d_model]
        v = self.v_proj(x)  # [b, s, d_model]

        q = rope_fn(q)  # [b, s, d_model]
        k = rope_fn(k)  # [b, s, d_model]

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)

        q = rearrange(q, "b s (g subh) d -> b g subh s d", g=self.num_groups)

        k = rearrange(k, "b s (g d) -> b g s d", g=self.num_groups)
        v = rearrange(v, "b s (g d) -> b g s d", g=self.num_groups)

        k_t = k.transpose(-2, -1)  # [b, G, d_kv, s]
        e = torch.einsum("b g subh s d, b g d s -> b g subh s s", q, k_t)
        e = e / math.sqrt(self.qdim)

        if attn_mask is not None:
            mask_5d = attn_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            mask_5d = mask_5d.expand(-1, self.num_groups, q.shape[2], s, -1)
            e = e.masked_fill(mask_5d == 0, float("-inf"))

        attn_weights = torch.softmax(e, dim=-1)

        out = torch.einsum("b g subh s s, b g s d -> b g subh s d", attn_weights, v)

        out = rearrange(out, "b g subh s d -> b s (g subh d)")

        out = self.out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, emb_dim: int, num_heads: int, num_groups: int):
        super().__init__()

        self.gqa = GQA(
            hidden_dim=emb_dim,
            num_heads=num_heads,
            num_groups=num_groups,
        )

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        attn_out = self.gqa(x)

        out = self.norm1(x + attn_out)
        out = self.mlp(out)

        return self.norm2(x + out)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_groups: int,
        num_heads: int,
        num_blocks: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, emb_dim)

        self.time_emb = nn.Linear(1, emb_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_groups=num_groups,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_proj = nn.Linear(emb_dim, latent_dim)

    def forward(self, latents: torch.Tensor, t: int):
        """
        latents: b, e
        t: b
        """
        t = rearrange(t, "b -> b 1")
        t_emb = self.time_emb(t)

        x = self.input_proj(latents)  # b e

        x = rearrange(x, "b e -> b 1 e")
        t_emb = rearrange(t_emb, "b d -> b 1 d")

        x = x + t_emb

        for block in self.blocks:
            x = block(x)

        out = self.output_proj(x)
        out = rearrange(out, "b 1 e -> b e")

        return out
