import torch
import os
from model.encoders.encoder_mae import mae_vit_base_patch16, mae_vit_large_patch16
from model.encoders.encoder_mae_group_channel import mae_vit_base_patch16_gc, mae_vit_large_patch16_gc
from model.encoders.encoder_vit import vit_base_patch16, vit_large_patch16
from model.encoders.encoder_vit_group_channel import vit_base_patch16_gc, vit_large_patch16_gc

# 768. 1024, 1280
def load_satvit_model(model_type, pretrained_root, **kwargs):
    if model_type == 'mae_base':
        model = mae_vit_base_patch16(**kwargs)
        feat_dim = 768
    elif model_type == 'mae_large':
        model = mae_vit_large_patch16(**kwargs)
        feat_dim = 1024
    elif model_type == 'mae_base_gc':
        model = mae_vit_base_patch16_gc(**kwargs)
        feat_dim = 768
    elif model_type == 'mae_large_gc':
        model = mae_vit_large_patch16_gc(**kwargs)
        feat_dim = 1024      
    if model_type == 'vit_base':
        model = vit_base_patch16(**kwargs)
        feat_dim = 768
    elif model_type == 'vit_large':
        model = vit_large_patch16(**kwargs)
        feat_dim = 1024
    elif model_type == 'vit_base_gc':
        model = vit_base_patch16_gc(**kwargs)
        feat_dim = 768
    elif model_type == 'vit_large_gc':
        model = vit_large_patch16_gc(**kwargs)
        feat_dim = 1024
    pretrained_path = os.path.join(pretrained_root, 'fmow_pretrain.pth')
    state_dict = torch.load(pretrained_path)
    model.load_state_dict(state_dict["model"])
    # optimizer
    return model, feat_dim
