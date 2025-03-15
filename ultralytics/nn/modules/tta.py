# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Test-Time Adaptation utilities."""

import torch
from torch.distributions import Normal

from .adaptor import LightweightAdaptor
from ultralytics.utils import LOGGER

__all__ = ("TTAMixin", "TTAStrategy")


class TTAMixin:
    """
    Test-Time Adaptation Mixin class - provides TTA capabilities to modules

    Can be used with any module that has feature outputs
    """

    def _init_tta(self, out_channels=None):
        """
        Initialize parameters and buffers needed for TTA

        Args:
            out_channels: Number of output channels, if None it should be set in the implementing class
        """
        self.register_buffer("_has_adaptor", torch.tensor(False))
        self.register_module("_adaptor", None)
        self._out_channels = out_channels

    @property
    def adaptor(self):
        """Return the test-time adaptor module"""
        return self._adaptor

    @adaptor.setter
    def adaptor(self, value):
        """Set the test-time adaptor module"""
        self._adaptor = value
        if value is not None:
            self._has_adaptor.fill_(True)
        else:
            self._has_adaptor.fill_(False)

    def add_adaptor(self, reduction_ratio=32, out_channels=None):
        """
        Add a lightweight adaptor to the module

        Args:
            reduction_ratio: Reduction ratio for the adaptor's intermediate layer channels
            out_channels: Optional, specify the number of output channels
        """
        if self.adaptor is None:
            channels = out_channels or self._out_channels
            if channels is None:
                raise ValueError("Output channels not specified, please provide out_channels parameter")

            self._adaptor = LightweightAdaptor(channels, reduction_ratio)
            self._has_adaptor.fill_(True)

    def apply_adaptor(self, x):
        """
        Apply the test-time adaptor (if present)

        Args:
            x: Input features

        Returns:
            Features processed by the adaptor
        """
        has_adaptor = getattr(self, "_has_adaptor", torch.tensor(False)).item()
        if has_adaptor and self._adaptor is not None:
            return self._adaptor(x)
        return x


class TTAStrategy:
    """
    Test-Time Adaptation Strategy - monitors distribution changes and dynamically adjusts model adaptability

    Decides when and how to update the adaptor by tracking feature statistics and loss changes
    """

    def __init__(
        self,
        model,
        alpha=0.01,
        tau1=1.1,
        tau2=1.05,
        momentum=0.99,
        reduction_ratio=32,
        tta_lr=0.001,
        feature_layer=-3,
    ):
        """
        Initialize TTA strategy

        Args:
            model: YOLO model
            alpha: EMA update rate
            tau1: Primary threshold
            tau2: Secondary threshold
            momentum: EMA momentum
            reduction_ratio: Reduction ratio for the adaptor
            tta_lr: Learning rate for TTA updates
            feature_layer: Index of the layer to extract features from
        """
        self.model = model
        self.alpha = alpha
        self.tau1 = tau1
        self.tau2 = tau2
        self.momentum = momentum
        self.reduction_ratio = reduction_ratio
        self.tta_lr = tta_lr
        self.feature_layer = feature_layer
        self.ema_mean = None
        self.ema_loss = None
        self.train_stats = {}
        self.current_stats = {}

    def extract_features(self, x):
        """
        Extract features from the specified layer and output detailed debug information

        Args:
            x: Input tensor

        Returns:
            Extracted features
        """
        features = None
        target_idx = len(self.model.model) + self.feature_layer

        LOGGER.info(f"Attempting to extract features from layer {target_idx}")

        for i, m in enumerate(self.model.model):
            x = m(x)
            LOGGER.info(f"Layer {i}: {m.__class__.__name__}, output shape {x.shape}")

            if i == target_idx:
                features = x
                LOGGER.info(f"Selected features from layer {i} with shape {features.shape}")
                break

        if features is None:
            LOGGER.warning(f"Target layer {target_idx} not found, using last layer output")
            features = x

        return features

    def collect_train_statistics(self, x):
        """
        Collect feature statistics from the training set

        Args:
            x: Input tensor or features
        """
        try:
            if x.dim() == 4 and x.size(1) == 3:
                features = self.extract_features(x)
            else:
                features = x

            if features is None:
                raise ValueError("Failed to extract features")

            LOGGER.info(f"Processing features with shape: {features.shape}")

            mu = torch.mean(features, dim=[0, 2, 3])
            sigma = torch.std(features, dim=[0, 2, 3])

            self.train_stats["mean"] = mu
            self.train_stats["std"] = sigma
            LOGGER.info(f"Train statistics collected: mean={mu.mean().item():.3f}, std={sigma.mean().item():.3f}")

        except Exception as e:
            LOGGER.error(f"Error collecting statistics: {str(e)}")
            raise

    def compute_distribution(self, x):
        """
        Compute feature distribution

        Args:
            x: Input tensor or features

        Returns:
            Tuple of mean and standard deviation
        """
        if x.dim() == 4 and x.size(1) == 3:
            features = self.extract_features(x)
        else:
            features = x

        mu = torch.mean(features, dim=[0, 2, 3])
        sigma = torch.std(features, dim=[0, 2, 3])

        self.current_stats["mean"] = mu
        self.current_stats["std"] = sigma

        return mu, sigma

    def update_ema(self, current_mean):
        """
        Update exponential moving average

        Args:
            current_mean: Current mean
        """
        if self.ema_mean is None:
            self.ema_mean = current_mean
        else:
            self.ema_mean = self.momentum * self.ema_mean + (1 - self.momentum) * current_mean

    def compute_domain_gap(self):
        """
        Compute domain gap

        Returns:
            KL divergence value
        """
        if not self.train_stats or not self.current_stats:
            return float("inf")

        kl_div = self.kl_divergence(
            self.train_stats["mean"], self.train_stats["std"], self.current_stats["mean"], self.current_stats["std"]
        )
        return kl_div

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate KL divergence between two distributions

        Args:
            mu1: Mean of first distribution
            sigma1: Standard deviation of first distribution
            mu2: Mean of second distribution
            sigma2: Standard deviation of second distribution

        Returns:
            KL divergence value
        """
        dist1 = Normal(mu1, sigma1)
        dist2 = Normal(mu2, sigma2)
        return torch.distributions.kl.kl_divergence(dist1, dist2).mean()

    def should_update(self, current_loss):
        """
        Determine whether an update is needed

        Args:
            current_loss: Current loss value

        Returns:
            Boolean value indicating if update is needed
        """
        if self.ema_loss is None:
            self.ema_loss = current_loss
            return True

        old_ema = self.ema_loss
        self.ema_loss = self.momentum * self.ema_loss + (1 - self.momentum) * current_loss

        if current_loss / old_ema > self.tau1:
            LOGGER.info(f"Major distribution shift detected: {current_loss / old_ema:.3f} > {self.tau1}")
            return True

        if current_loss / old_ema > self.tau2:
            LOGGER.info(f"Minor distribution shift detected: {current_loss / old_ema:.3f} > {self.tau2}")
            return True

        return False

    def init_adaptors(self, model=None):
        """
        Initialize adaptors in the model

        Args:
            model: Optional model to initialize

        Returns:
            Number of initialized adaptors
        """
        from .block import Bottleneck

        target_model = model if model is not None else self.model
        adaptor_count = 0

        for module in target_model.modules():
            if isinstance(module, Bottleneck):
                if not hasattr(module, "add_adaptor"):
                    LOGGER.warning(f"Module {module.__class__.__name__} does not support TTA")
                    continue

                module.add_adaptor(reduction_ratio=self.reduction_ratio)
                adaptor_count += 1

        LOGGER.info(f"Initialized {adaptor_count} TTA adaptors")
        return adaptor_count
