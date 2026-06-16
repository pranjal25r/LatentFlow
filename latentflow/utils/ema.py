"""Exponential Moving Average (EMA) for model weight smoothing."""

from typing import Iterator
import torch
import torch.nn as nn
from torch import Tensor


class EMA:
    """Exponential Moving Average for model weights.
    
    Maintains a smoothed copy of model parameters. Used during training to
    improve generalization and sampling quality.
    
    Args:
        model: Model to track.
        decay: Decay rate (typically 0.9999). Higher = slower updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Initialize EMA tracker."""
        self.model = model
        self.decay = decay
        self.ema_model = self._create_ema_model()

    def _create_ema_model(self) -> nn.Module:
        """Create a copy of the model for EMA."""
        ema_model = type(self.model)(*self._get_init_args())
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.requires_grad_(False)
        return ema_model

    def _get_init_args(self) -> tuple:
        """Extract initialization arguments from model.
        
        This is a fallback that tries common patterns. For custom models,
        override this method or manually set ema_model.
        """
        # Try to extract from model config if available
        if hasattr(self.model, "latent_channels"):
            return (
                self.model.latent_channels,
                self.model.hidden_size,
                self.model.depth,
                self.model.num_heads,
                self.model.mlp_ratio,
                self.model.patch_size,
                self.model.dropout,
            )
        raise RuntimeError(
            "Cannot auto-detect model init args. "
            "Please manually create ema_model and set it."
        )

    def update(self) -> None:
        """Update EMA model with current model weights."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters(),
            ):
                # ema_param = decay * ema_param + (1 - decay) * model_param
                ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)

    def state_dict(self) -> dict:
        """Get EMA model state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)

    def eval(self) -> nn.Module:
        """Set EMA model to eval mode and return it."""
        self.ema_model.eval()
        return self.ema_model
