import argparse
import logging
import math
import os
import random
import sys
import copy
import cv2
import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
import options as option
from model.bk_RayDiffusionModel import RayDiffuser
from model.scheduler import NoiseScheduler
from torch.cuda.amp import autocast
sys.path.insert(0, "../../")
import utils as utils
from datasets.satellite import SatelliteDataset
# torch.autograd.set_detect_anomaly(True)
import model.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
# from accelerate import Accelerator
import tqdm
device = "cuda:0"


rescale_fn = {
    "zero": lambda x: 0,
    "identity": lambda x: x,
    "square": lambda x: x**2,
    "square_root": lambda x: torch.sqrt(x),
}

class Trainer:
    def __init__(self, opt):
        self.device = device
        self.opt = opt
        self.feature_type = "mae_large"
        if opt["training"].get("pretrain_path", None):
            self.resume()
        else:
            #### mkdir and loggers
            #### Predictor path
            utils.mkdir_and_rename(
                opt["training"]["experiments_root"]
            )  # rename experiment folder if exists

            self.resume_state = False
            # config loggers. Before it, the log will not work
            utils.setup_logger(
                "base",
                opt["training"]["experiments_root"],
                "train_" + opt["name"],
                level=logging.INFO,
                screen=False,
                tofile=True,
            )
            utils.setup_logger(
                "val",
                opt["training"]["experiments_root"],
                "val_" + opt["name"],
                level=logging.INFO,
                screen=False,
                tofile=True,
            )
            self.logger = logging.getLogger("base")
            self.logger.info(option.dict2str(opt))
            # tensorboard logger

            self.tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))

        # # accelerator
        # self.accelerator = Accelerator(
        #     split_batches = True, # split_batches,
        #     mixed_precision = 'no'
        # )

        # self.accelerator.native_amp = amp

        # prepare the dataloaders
        train_set = SatelliteDataset(opt["dataset"]["root"], split='train')
        self.train_size = int(math.ceil(len(train_set) / opt["training"]["batch_size"]))
        self.total_iters = int(opt["training"]["max_iterations"])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))

        train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt["training"]["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        )

        self.logger.info(
            "Number of train images: {:,d}, iters: {:,d}".format(
                len(train_set), self.train_size
            )
        )
        self.logger.info(
            "Total epochs needed: {:d} for iters {:,d}".format(
                self.total_epochs, self.total_iters
            )
        )
        # the val set
        val_set =  SatelliteDataset(opt["dataset"]["root"], split='val')
        val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=opt["val"]["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        )
        self.logger.info(
            "Number of val images in [{:s}]: {:d}".format(
                opt["dataset"]["name"], len(val_set)
            )
        )
        assert train_loader is not None
        assert val_loader is not None
        self.train_loader = train_loader
        self.val_loader = val_loader   

        #### create model
        self.model = RayDiffuser(       
            model_type="dit",
            depth=8,
            width=16,
            hidden_size=1152,
            P=1,
            max_num_images=1,
            feature_extractor="mae_large",
            pretrain_root="/root/autodl-pub/ZTT/pretrained/satmae"
            # pretrain_root="/root/autodl-pub/ZTT/pretrained/dinov2/ckpt",
            ).to(self.device)
        
        self.noise_scheduler = NoiseScheduler(
        type=opt["noise_scheduler"]["type"],
        max_timesteps=opt["noise_scheduler"]["max_timesteps"],
        beta_start=opt["noise_scheduler"]["beta_start"],
        beta_end=opt["noise_scheduler"]["beta_end"],
        )
        optim_params = []
        for (
            k,
            v,
        ) in self.model.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:

                self.logger.warning("Params [{:s}] will not optimize.".format(k))

        # set optimizers        
        self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=opt["training"]["lr"],
                    weight_decay=opt["training"]["weight_decay"],
                    betas=(opt["training"]["beta1"], opt["training"]["beta2"]),
                )
        self.scheduler = lr_scheduler.MultiStepLR_Restart(
                        self.optimizer,
                        opt["training"]["lr_steps"],
                        restarts=opt["training"]["restarts"],
                        weights=opt["training"]["restart_weights"],
                        gamma=opt["training"]["lr_gamma"],
                        clear_state=opt["training"]["clear_state"],
                    )  
                

    def resume(self):
        device_id = torch.cuda.current_device()
        try:
            resume_state = torch.load(
                self.opt["training"]["pretrain_path"],
                map_location=lambda storage, loc: storage.cuda(device_id),
            )
            option.check_resume(self.opt, resume_state["iter"])  # check resume options

            self.resume_state = True
            print('resuming from {}'.format(self.opt["training"]["pretrain_path"]))
        except:
            self.resume_state = False
            print('from scratch')
        
    def train(self):
        #### resume training
        if self.resume_state:
            self.logger.info(
                "Resuming training from epoch: {}, iter: {}.".format(
                    self.resume_state["epoch"], self.resume_state["iter"]
                )
            )

            start_epoch = self.resume_state["epoch"]
            current_step = self.resume_state["iter"]
            self.model.resume_training(self.resume_state)  # handle optimizers and schedulers
        else:
            current_step = 0
            start_epoch = 0
        
            #### training
        self.logger.info(
            "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
        )

        # best_psnr = 0.0
        # best_iter = 0

        # -------------------------------------------------------------------------
        # ------------------------- training starts here --------------------------
        # -------------------------------------------------------------------------
        for epoch in range(start_epoch, self.total_epochs + 1):
            psnr_ls = []
            total_loss = 0.
            for idx, batch in enumerate(self.train_loader):
                current_step += 1

                if current_step > self.total_iters:
                    break          

                ret = self.forward_rays_iteration(batch)
                loss_dict = self.cal_losses(ret, batch)
                loss = loss_dict["all"]
                total_loss += loss.item()
                psnr_ls.append(loss_dict["psnr"])
                
                loss.backward()
                # self.accelerator.backward(loss)

                    
                # self.accelerator.wait_for_everyone()
                # self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimiz

                if current_step % self.opt["logger"]["print_freq"] == 0:
                    print(f'loss: {total_loss:.4f},  psnr: {psnr_mean:.4f}')
                    # logs = model.get_current_log()
                    # message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    #     epoch, current_step, model.get_current_learning_rate()
                    # )
                    # for k, v in logs.items():
                    #     message += "{:s}: {:.4e} ".format(k, v)
                    #     # tensorboard logger
                    #     self.tb_logger.add_scalar(k, v, current_step)

                    # self.logger.info(message)

                # validation, to produce ker_map_list(fake)
                if current_step % self.opt["training"]["val_freq"] == 0:  
                    self.val()

                #### save models and training states
                if current_step % self.opt["logger"]["save_checkpoint_freq"] == 0:
                    self.logger.info("Saving models and training states.")
                    self.save(current_step)
            psnr_mean = np.mean(np.array(psnr_ls))
        self.logger.info("Saving the final model.")
        self.save("latest")
        self.logger.info("End of Predictor and Corrector training.")
        self.tb_logger.close()

    def forward_rays_iteration(self, 
                          batch,
                          visualize=False,
                          pred_x0=True,
                          clip_bounds_m=5,
                          clip_bounds_d=5,
                          normalize_directions=True,
                          normalize_moments=True,
                          rescale_noise="square_root",
                          beta_tilde=False,
                          return_intermediate=False):
        pbar = False
        stop_iteration = -1
        images = batch['image'].to(self.device).squeeze()
        if self.feature_type == 'dinov2':
            image_features = self.model.feature_extractor(images, autoresize=True)
        elif self.feature_type == 'mae_large':
            image_features = self.model.feature_extractor(images)
        
        num_train_steps = self.model.noise_scheduler.max_timesteps
        ray_batch_size = images.shape[-2] * images.shape[-1]
        num_images = images.shape[0]

        x_t = torch.randn(
            ray_batch_size, num_images, 6, device=device
        )
        if visualize:
            x_ts = [x_t]
            all_pred = []
            noise_samples = []
        
        loop = range(num_train_steps - 1, stop_iteration, -1)
        if pbar:
            loop = tqdm(loop)
        for t in loop:
            z = (
                torch.randn(
                    batch_size,
                    num_images,
                    6,
                    device=device,
                )
                if t > 0
                else 0
            )
            # eps_pred, noise_sample = self.model(
            #     features=image_features,
            #     rays_noisy=x_t,
            #     t=t,
            #     compute_x0=compute_x0,
            # )
            x_noise, eps_pred = self.model(
                image_features, 
                t, 
                epsilon=None, 
                mask=None
            )

            if pred_x0:
                c = torch.linalg.norm(eps_pred[:, :, :3], dim=2, keepdim=True)
                d = eps_pred[:, :, :3]
                m = eps_pred[:, :, 3:]
                if normalize_directions:
                    eps_pred[:, :, :3] = d / c
                if normalize_moments:
                    eps_pred[:, :, 3:] = m - (d * m).sum(dim=2, keepdim=True) * d / c

            if visualize:
                all_pred.append(eps_pred.clone())
                noise_samples.append(noise_sample)

            alpha_t = self.model.noise_scheduler.alphas[t]
            alpha_bar_t = self.model.noise_scheduler.alphas_cumprod[t]
            if t == 0:
                alpha_bar_tm1 = torch.ones_like(alpha_bar_t)
            else:
                alpha_bar_tm1 = self.model.noise_scheduler.alphas_cumprod[t - 1]
            beta_t = self.model.noise_scheduler.betas[t]
            if beta_tilde:
                sigma_t = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
            else:
                sigma_t = beta_t

            sigma_t = rescale_fn[rescale_noise](sigma_t)

            if pred_x0:
                w_x0 = torch.sqrt(alpha_bar_tm1) * beta_t / (1 - alpha_bar_t)
                w_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
                x_t = w_x0 * eps_pred + w_xt * x_t + sigma_t * z
            else:
                scaled_noise = sigma_t / torch.sqrt(1 - alpha_bar_t) * eps_pred
                x_t = (x_t - scaled_noise) / torch.sqrt(alpha_t) + sigma_t * z

            x_t_d = torch.clip(x_t[..., :3], min=-1 * clip_bounds_m, max=clip_bounds_m)
            x_t_m = torch.clip(x_t[..., 3:], min=-1 * clip_bounds_d, max=clip_bounds_d)
            x_t = torch.cat((x_t_d, x_t_m), dim=-1)

            if visualize:
                x_ts.append(x_t.detach().clone())

        # rays_final, rays_intermediate, pred_intermediate, _ = x_t, x_ts, all_pred, noise_samples
        if return_intermediate:
            return {
                "rays_final": x_t,
                "rays_intermediate": x_ts,
                "pred_intermediate": all_pred,         
            }
        else:
            return {
                "rays_final": x_t}

    def cal_losses(self, ret, batch):
        rays = ret["rays_final"]
        batch = None

    def val(self):
        avg_psnr = 0.0
        idx = 0
        for it, batch in enumerate(self.val_loader):

            # valid Predictor

            # calculate PSNR
            avg_psnr += utils.calculate_psnr(output, gt_img)
            idx += 1

        avg_psnr = avg_psnr / idx

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_iter = current_step

        # log
        self.logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
        logger_val = logging.getLogger("val")  # validation logger
        logger_val.info(
            "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                epoch, current_step, avg_psnr
            )
        )
        print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                epoch, current_step, avg_psnr
            ))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger.add_scalar("psnr", avg_psnr, current_step)


if __name__ == '__main__':
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="conf/wv3.yaml")
    # parser.add_argument(
    #     "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
    # )
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(opt)
    trainer.train()
    


