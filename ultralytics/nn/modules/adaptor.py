# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Adaptor module."""

import torch.nn as nn

__all__ = "LightweightAdaptor"


class LightweightAdaptor(nn.Module):
    """A lightweight neural network adaptor module for feature refinement."""

    def __init__(
        self,
        in_channels,
        reduction_ratio=32,
        out_channels=None,
        act_type="silu",
        use_bn=True,
        scale_residual=1.0,
        init_values=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.scale_residual = scale_residual
        hidden_channels = max(1, in_channels // reduction_ratio)

        # Projection layer for shortcut connection (when channels don't match)
        self.proj = None
        if self.in_channels != self.out_channels:
            self.proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=True)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        # Down projection layer
        self.down_proj = nn.Conv2d(self.in_channels, hidden_channels, 1)

        # Normalization layer
        self.norm = nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity()

        # Activation function
        if act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_type}")

        # Up projection layer
        self.up_proj = nn.Conv2d(hidden_channels, self.out_channels, 1)

        # Initialization
        nn.init.kaiming_normal_(self.down_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.down_proj.bias)
        nn.init.constant_(self.up_proj.weight, init_values)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.down_proj(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.up_proj(out)

        # Apply residual scaling
        if self.scale_residual != 1.0:
            out = out * self.scale_residual

        return identity + out
