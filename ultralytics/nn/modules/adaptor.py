# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adaptor modules."""

import torch.nn as nn
from .conv import Conv

__all__ = "Adaptor"


class Adaptor(nn.Module):
    """
    Lightweight Adaptor module for test-time adaptation.

    Introduced in https://arxiv.org/abs/2312.17461. This module is added in parallel
    to backbone blocks and only its parameters are updated during test-time adaptation.
    The output of this module is the original block's output plus the adaptation adjustment.

    Args:
        c1 (int): Input channels (dimension d).
        r (int, optional): Channel reduction ratio. Defaults to 32.
        use_conv (bool, optional): Whether to use Conv layers (for CNN blocks) or Linear layers (for Transformer blocks).
                                   Defaults to True (using Conv).
    """

    def __init__(self, c1, r=32, use_conv=True):
        """Initializes the Adaptor module."""
        super().__init__()
        self.c_ = max(1, c1 // r)
        self.use_conv = use_conv
        if self.use_conv:
            self.down_proj = Conv(c1, self.c_, k=1, act=False)  # W_down
            self.up_proj = Conv(self.c_, c1, k=1, act=False)  # W_up
        else:
            self.down_proj = nn.Linear(c1, self.c_, bias=False)  # W_down
            self.up_proj = nn.Linear(self.c_, c1, bias=False)  # W_up
        self.act = nn.ReLU()

        if self.use_conv:
            nn.init.zeros_(self.up_proj.conv.weight)
            if hasattr(self.up_proj, "bn") and self.up_proj.bn.bias is not None:
                nn.init.zeros_(self.up_proj.bn.bias)
            elif self.up_proj.conv.bias is not None:
                nn.init.zeros_(self.up_proj.conv.bias)
        else:
            nn.init.zeros_(self.up_proj.weight)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Forward pass of the Adaptor module. Calculates the adaptation adjustment
        and adds it to the original input tensor (residual connection).

        Args:
            x (torch.Tensor): Input tensor from the backbone block (identity).

        Returns:
            (torch.Tensor): Output tensor (identity + adaptation adjustment).
        """
        identity = x

        if self.use_conv:
            x_adapt = self.down_proj(x)
            x_adapt = self.act(x_adapt)
            x_adapt = self.up_proj(x_adapt)
        else:
            original_shape = x.shape
            if len(original_shape) > 2:
                x_flat = x.view(-1, original_shape[-1])
                x_adapt_flat = self.down_proj(x_flat)
                x_adapt_flat = self.act(x_adapt_flat)
                x_adapt_flat = self.up_proj(x_adapt_flat)
                x_adapt = x_adapt_flat.view(*original_shape)
            else:
                x_adapt = self.down_proj(x)
                x_adapt = self.act(x_adapt)
                x_adapt = self.up_proj(x_adapt)

        return identity + x_adapt
