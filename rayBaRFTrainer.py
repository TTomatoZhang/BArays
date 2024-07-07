import os
import time
import numpy as np
import shutil

import torch
import torch.utils.data.distributed
import torch.distributed as dist
from baseTrainer import BaseTrainer
# from models.geometry.depth import inv2depth
from models.rayBaRF import rayBaRF
from models.render_image import render_single_image
from models.sample_ray import RaySamplerSingleImage
from visualization.cam_visualizer import *
from models.visualization.feature_visualizer import *
from utils.img_utils import img2mse, mse2psnr, img_HWC2CHW, img2psnr, colorize
from utils.sat_utils import projection
from utils.render_utils import render_rays
import models.config as config
from models.loss.criterion import MaskedL2ImageLoss


class rayBaRFTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.state = 'rays_only'

    def build_networks(self):
        self.model = rayBaRF(self.config,
                                load_opt=not self.config.no_load_opt,
                                load_scheduler=not self.config.no_load_scheduler,
                                pretrained=self.config.pretrained)

    def setup_optimizer(self):
        # the basic optimizers
        learnable_params = list(self.model.nerf.parameters())
        learnable_params += list(self.model.feature_net.parameters())
        if self.model.net_fine is not None:
            learnable_params += list(self.model.net_fine.parameters())

        if self.model.net_fine is not None:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.nerf.parameters()},
                {'params': self.model.feature_net.parameters(), 'lr': self.config.lrate_feature}],
                lr=self.config.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.nerf.parameters()},
                {'params': self.model.feature_net.parameters(), 'lr': self.config.lrate_feature}],
                lr=self.config.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config.lrate_decay_steps,
                                                         gamma=self.config.lrate_decay_factor)        

        # the extra optimizers
        self.depth_optimizer = torch.optim.Adam([
            dict(params=self.model.depth_learner.parameters(), lr=self.args.lrate_rays)
        ])

        self.depth_scheduler = torch.optim.lr_scheduler.StepLR(
            self.depth_optimizer, step_size=self.args.lrate_decay_rays_steps, gamma=0.5)

    def setup_loss_functions(self):
        # basic 
        self.rgb_loss = MaskedL2ImageLoss()

    def compose_state_dicts(self) -> None:
        # the basic 
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict()}
        
        self.state_dicts['models']['nerf'] = self.model.nerf
        self.state_dicts['models']['feature_net'] = self.model.feature_net

        self.state_dicts['optimizers']['optimizer'] = self.optimizer
        self.state_dicts['schedulers']['scheduler'] = self.scheduler        
        # the extra
        self.state_dicts['models']['depth_learner'] = self.model.depth_learner
        self.state_dicts['optimizers']['depth_optimizer'] = self.depth_optimizer
        self.state_dicts['schedulers']['depth_scheduler'] = self.depth_scheduler

    def train_iteration(self, data_batch) -> None:
        ######################### 3-stages training #######################
        # ---- (1) Train the depth optimizer with self-supervised loss.<---|
        # |             (10000 iterations)                                |
        # |--> (2) Train ibrnet while fixing the depth optimizer.          |
        # |             (10000 iterations)                                |
        # |--> (3) Jointly train the depth optimizer and ibrnet.           |
        # |             (10000 iterations)                                |
        # |-------------------------->------------------------------------|
        if self.iteration % 10000 == 0 and (self.iteration // 10000) % 2 == 0:
            self.state = self.model.switch_state_machine(state='rays_only')
        elif self.iteration % 10000 == 0 and (self.iteration // 10000) % 2 == 1:
            self.state = self.model.switch_state_machine(state='nerf_only')
        if self.iteration != 0 and self.iteration % 30000 == 0:
            self.state = self.model.switch_state_machine(state='joint')

        images = torch.cat([data_batch['rgb'], data_batch['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
        all_feat_maps = self.model.feature_net(images)

        feat_maps = (all_feat_maps[0][1:, :32, ...], None) 

        depth_feats = all_feat_maps[0]

        min_depth, max_depth = data_batch['depth_range'][0][0], data_batch['depth_range'][0][1]


        # load training rays
        ray_sampler = RaySamplerSingleImage(data_batch, self.device)
        N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / data_batch['src_rgbs'][0].shape[0])

        ray_batch = ray_sampler.random_sample(N_rand,
                                              sample_mode=self.args.sample_mode,
                                              center_ratio=self.args.center_ratio,
                                              )
        # Start of core optimization loop
        pred_depths, pred_rel_rays, sfm_loss, fmap = self.model.correct_rays(
            fmaps=depth_feats,
            target_image=data_batch['rgb'].cuda(),
            ref_imgs=data_batch['src_rgbs'].cuda(),
            rpcs=data_batch['rpcs'],
            ray_batch=ray_batch,
            min_depth=min_depth,
            max_depth=max_depth,
            scaled_shape=data_batch['scaled_shape'],
            )

        # The predicted depth is used as a weak supervision to NeRF.
        self.pred_depths = pred_depths[-1]
        depth_prior = pred_depths[-1].detach().clone()
        depth_prior = depth_prior.reshape(-1, 1)[ray_batch['selected_inds']]

        ret = render_rays()

        loss_all = 0
        loss_dict = {}

        # compute loss
        self.optimizer.zero_grad()
        self.depth_optimizer.zero_grad()

        if self.state == 'rays_only' or self.state == 'joint':
            loss_dict['sfm_loss'] = sfm_loss['loss']
            self.scalars_to_log['loss/photometric_loss'] = sfm_loss['metrics']['photometric_loss']
            if 'smoothness_loss' in sfm_loss['metrics']:
                self.scalars_to_log['loss/smoothness_loss'] = sfm_loss['metrics']['smoothness_loss']

        rgb_loss = self.rgb_loss(ret['rgb'], ray_batch)

        loss_dict['nerf_loss'] = rgb_loss

            
        if self.state == 'joint':
            # loss_all += loss_depth.item()
            loss_all += self.model.compose_joint_loss(
                loss_dict['sfm_loss'], loss_dict['nerf_loss'], self.iteration)
        elif self.state == 'rays_only':
            loss_all += loss_dict['sfm_loss']
        else: # nerf_only
            # loss_all += loss_depth.item()
            loss_all += loss_dict['nerf_loss']

        # with torch.autograd.detect_anomaly():
        loss_all.backward()

        if self.state == 'rays_only' or self.state == 'joint':
            self.depth_optimizer.step()
            self.depth_scheduler.step()

        if self.state == 'nerf_only' or self.state == 'joint':
            self.optimizer.step()
            self.scheduler.step()

        if self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['rgb'], ray_batch['rgb']).item()
            self.scalars_to_log['train/mse_loss'] = mse_error
            self.scalars_to_log['train/psnr-training-batch'] = mse2psnr(mse_error)
            self.scalars_to_log['loss/final'] = loss_all.item()
            self.scalars_to_log['loss/rgb_loss'] = rgb_loss

            self.scalars_to_log['lr/IBRNet'] = self.scheduler.get_last_lr()[0]
            self.scalars_to_log['lr/depth'] = self.depth_scheduler.get_last_lr()[0]
      
    def validate(self) -> float:
        self.model.switch_to_eval()

        target_image = self.train_data['rgb'].squeeze(0).permute(2, 0, 1)
        pred_depths_gray = self.pred_depths.squeeze(0).detach().cpu()
        pred_depths = self.pred_depths.squeeze(0).squeeze(0)
        pred_depth_color = colorize(pred_depths.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)

        self.writer.add_image('train/target_image', target_image, self.iteration)
        self.writer.add_image('train/pred_depths', pred_depths_gray, self.iteration)
        self.writer.add_image('train/pred_depth-color', pred_depth_color, self.iteration)

        # Logging a random validation view.
        val_data = next(self.val_loader_iterator)
        tmp_ray_sampler = RaySamplerSingleImage(val_data, self.device, render_stride=self.config.render_stride, \
                                                sim3=None)

        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
        score = log_view_to_tb(
            self.writer, self.iteration, self.config, self.model, tmp_ray_sampler,
            gt_img, render_stride=self.config.render_stride, prefix='val/',
            data=val_data, dataset=self.val_dataset)

        # Logging current training view.
        tmp_ray_train_sampler = RaySamplerSingleImage(self.train_data, self.device, render_stride=1)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
        log_view_to_tb(self.writer, self.iteration, self.config, self.model, tmp_ray_train_sampler, 
                       gt_img, render_stride=1, prefix='train/', data=self.train_data, dataset=self.train_dataset)

        torch.cuda.empty_cache()
        self.model.switch_to_train()

        return score


@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, ray_sampler, gt_img,
                   render_stride=1, prefix='', data=None, dataset=None) -> float:
    # with torch.no_grad():
    ray_batch = ray_sampler.get_all()
    if model.feature_net is not None:
        images = torch.cat([data['rgb'], data['src_rgbs'].squeeze(0)], dim=0).cuda().permute(0, 3, 1, 2)
        all_feat_maps = model.feature_net(images)
        # depth_feats = all_feat_maps[2][:, ...]
        feat_maps = (all_feat_maps[0][1:, :32, ...], None) if model.net_fine is None else \
                    (all_feat_maps[0][1:, :32, ...], all_feat_maps[1][1:, ...])
        # feat_maps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
    else:
        feat_maps = [None, None]

    # depths, rel_rays, rel_rpcs, sfm_loss, fmap
    pred_depths, pred_rel_rays, _, __ = model.correct_rays(
                            fmaps=all_feat_maps[0],
                            target_image=data['rgb'].cuda(),
                            ref_imgs=data['src_rgbs'].cuda(),
                            min_depth=data['depth_range'][0][0],
                            max_depth=data['depth_range'][0][1],
                            scaled_shape=data['scaled_shape'])
    depth_prior = pred_depths.reshape(-1, 1).detach().clone()

    if prefix == 'val/':
        pred_depths = pred_depths.squeeze(0).squeeze(0)
        pred_depths = colorize(pred_depths.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)
        writer.add_image(prefix + 'pred_depths', pred_depths, global_step)

    ret = render_single_image()
    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['depth'].detach().cpu()
    acc_map = torch.sum(ret['weights'], dim=-1).detach().cpu()


    depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
    acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt', depth_im, global_step)
    writer.add_image(prefix + 'acc', acc_map, global_step)

    # plot_feature_map(writer, global_step, ray_sampler, feat_maps, prefix)

    # write scalar
    pred_rgb = ret['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    return psnr_curr_img

