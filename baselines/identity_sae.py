import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    context_size: int = 128  # Can be used for auto-interp
    hook_head_index: Optional[int] = None


class IdentitySAE(nn.Module):
    def __init__(self, model_name: str, d_model: int, hook_layer: int):
        super().__init__()

        # Initialize W_enc and W_dec as identity matrices
        self.W_enc = nn.Parameter(torch.eye(d_model))
        self.W_dec = nn.Parameter(torch.eye(d_model))

        hook_name = f"blocks.{hook_layer}.hook_resid_post"

        # Initialize the configuration dataclass
        self.cfg = SAEConfig(
            model_name, d_in=d_model, d_sae=d_model, hook_name=hook_name, hook_layer=hook_layer
        )

    def encode(self, input_acts: torch.Tensor):
        acts = input_acts @ self.W_enc
        return acts

    def decode(self, acts: torch.Tensor):
        return acts @ self.W_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon
