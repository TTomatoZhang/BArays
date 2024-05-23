import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from model.dit import DiT
from model.encoders.feature_extractors_dino import SpatialDino
from model.encoders import load_satvit_model
from model.scheduler import NoiseScheduler


class RayDiffuser(nn.Module):
    def __init__(
        self,
        objective='train',
        model_type="dit",
        depth=8,
        width=16,
        hidden_size=1152,
        P=1,
        max_num_images=1,
        noise_scheduler=None,
        freeze_encoder=True,
        feature_extractor="dino",
        use_unconditional=False,
        pretrain_root="/root/autodl-pub/ZTT/pretrained/dinov2/ckpt",
    ):
        super().__init__()
        if noise_scheduler is None:
            self.noise_scheduler = NoiseScheduler()
        else:
            self.noise_scheduler = noise_scheduler

        self.objective = objective
        self.ray_dim = 6

        self.width = width

        self.max_num_images = max_num_images
        self.model_type = model_type
        self.use_unconditional = use_unconditional

        if feature_extractor == "dino":
            self.feature_extractor = SpatialDino(
                freeze_weights=freeze_encoder, 
                model_root=pretrain_root,
                num_patches_x=width, 
                num_patches_y=width
            )
            self.feature_dim = self.feature_extractor.feature_dim
        elif "mae" in feature_extractor or "vit" in feature_extractor:
            self.feature_extractor, self.feature_dim = load_satvit_model(feature_extractor, pretrained_root=pretrain_root)
        
        else:
            raise Exception(f"Unknown feature extractor {feature_extractor}")

        if self.use_unconditional:
            self.register_parameter(
                "null_token", nn.Parameter(torch.randn(self.feature_dim, 1, 1))
            )

        self.input_dim = self.ray_dim + self.feature_dim

        if model_type == "dit":
            self.ray_predictor = DiT(
                in_channels=self.input_dim,
                out_channels=self.ray_dim,
                width=width,
                depth=depth,
                hidden_size=hidden_size,
                max_num_images=max_num_images,
                P=P,
            ) 
        else:
            raise Exception(f"Unknown model type {model_type}")

    def from_x0_to_noise(self, x, t, epsilon=None, mask=None):
        """
        forward diffusion (adds noise) to the input.

        If a mask is provided, the noise is only applied to the masked inputs.
        """
        t = t.reshape(-1, 1, 1, 1, 1)
        if epsilon is None:
            epsilon = torch.randn_like(x)
        else:
            epsilon = epsilon.reshape(x.shape)
        alpha_bar = self.noise_scheduler.alphas_cumprod[t]
        x_noise = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * epsilon
        if mask is not None:
            x_noise = x_noise * mask + x * (1 - mask)
        return x_noise, epsilon

    def from_noise_to_result()

    def forward_(
        self,
        features=None,
        images=None,
        rays=None,
        rays_noisy=None,
        t=None,
        mask=None,
        unconditional_mask=None,
    ):
        """
        Args:
            images: (B, N, 3, H, W).
            t: (B,).
            rays: (B, N, 6, H, W).
            rays_noisy: (B, N, 6, H, W).
            unconditional_mask: (B, N) or (B,). Should be 1 for unconditional samples
                and 0 else.
        """

        if features is None:
            features = self.feature_extractor(images, autoresize=False)

        B = features.shape[0]

        if unconditional_mask is not None and self.use_unconditional:
            null_token = self.null_token.reshape(1, 1, self.feature_dim, 1, 1)
            unconditional_mask = unconditional_mask.reshape(B, -1, 1, 1, 1)
            features = (
                features * (1 - unconditional_mask) + null_token * unconditional_mask
            )

        if isinstance(t, int) or isinstance(t, np.int64):
            t = torch.ones(1, dtype=int).to(features.device) * t
        else:
            t = t.reshape(B)

        if rays_noisy is None:
            rays_noisy, epsilon = self.forward_noise(rays, t, mask=mask)
        else:
            epsilon = None

        scene_features = torch.cat([features, rays_noisy], dim=2)

        epsilon_pred = self.ray_predictor(scene_features, t)

        if self.objective != 'train':
            t = t.reshape(-1, 1, 1, 1, 1)
            a = self.noise_scheduler.alphas_cumprod[t]
            x0 = (rays_noisy - torch.sqrt(1 - a) * epsilon_pred) / torch.sqrt(a)
            return epsilon_pred, x0
        else:
            return epsilon_pred, epsilon
    
    def forward(self,
                images,
                t, 
                rays=None,
                rays_noise=None,
                ):
        '''        
        Args:
            images: (B, 3, H, W).
            t: (B,).
            rays: (B, N, 6, H, W).
            rays_noisy: (B, N, 6, H, W).
        '''

        features = self.feature_extractor(images, autoresize=False)

        B = features.shape[0]
        
        if isinstance(t, int) or isinstance(t, np.int64):
            t = torch.ones(1, dtype=int).to(features.device) * t
        else:
            t = t.reshape(B)
            
        if self.objective == 'train':
            assert rays is not None
            rays_noise, epsilon = self.forward_noise(rays, t, mask=None)
            scene_features = torch.cat([features, rays_noise], dim=2)
            epsilon_pred = self.ray_predictor(scene_features, t)
            return epsilon_pred, epsilon
        else:
            assert rays_noise is not None
            epsilon = None

        
            
            