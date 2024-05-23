import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from model.dit import DiT
from model.encoders.feature_extractors_dino import SpatialDino
from model.encoders import load_satvit_model
from model.scheduler import NoiseScheduler
from model.diffuser import GaussianDiffusion
from model.denoiser import Denoiser

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

        self.ray_dim = 6

        self.width = width


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


        self.input_dim = self.ray_dim + self.feature_dim

        self.diffuser = GaussianDiffusion()
        self.diffuser.model = Denoiser()

        self.target_dim = self.diffuser.model.target_dim

        self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

    def forward(
        self,
        image,
        gt_rays = None,
        training=True,
    ):
        z = self.feature_extractor(image)  
        ret = self.diffuser(pose_encoding, z=z)
        ret = {
            "pred_cameras": ,
        }
        return ret