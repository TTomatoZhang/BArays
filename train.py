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
from model.diffuser import RayDiffuser
from ray_diffusion.model.scheduler import NoiseScheduler
from torch.cuda.amp import autocast
sys.path.insert(0, "../../")
import utils as util
from datasets.satellite import SatelliteDataset
# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from accelerate import Accelerator

device = "cuda:0"

class Trainer:
    def __init__(self, opt):
        self.device = device
        self.opt = opt
        if opt["training"].get("pretrain_path", None):
            self.resume
        else:
            #### mkdir and loggers
            # Predictor path
            util.mkdir_and_rename(
                opt["training"]["experiments_root"]
            )  # rename experiment folder if exists

            
            # config loggers. Before it, the log will not work
            util.setup_logger(
                "base",
                opt["training"]["experiments_root"],
                "train_" + opt["name"],
                level=logging.INFO,
                screen=False,
                tofile=True,
            )
            util.setup_logger(
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

        # accelerator
        self.accelerator = Accelerator(
            split_batches = True, # split_batches,
            mixed_precision = 'no'
        )

        self.accelerator.native_amp = amp

        # prepare the dataloaders
        train_set = SatelliteDataset(opt["dataset"]["root"], split='train')
        train_size = int(math.ceil(len(train_set) / opt["trainining"]["batch_size"]))
        total_iters = int(opt["trainining"]["max_iterations"])
        self.total_epochs = int(math.ceil(total_iters / train_size))

        train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt["trainining"]["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        )

        self.logger.info(
            "Number of train images: {:,d}, iters: {:,d}".format(
                len(train_set), train_size
            )
        )
        self.logger.info(
            "Total epochs needed: {:d} for iters {:,d}".format(
                self.total_epochs, total_iters
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
            max_num_images=1).to(self.device)
        
        self.noise_scheduler = NoiseScheduler(
        type=opt["noise_scheduler"]["type"],
        max_timesteps=opt["noise_scheduler"]["max_timesteps"],
        beta_start=opt["noise_scheduler"]["beta_start"],
        beta_end=opt["noise_scheduler"]["beta_end"],
    )

    def resume(self):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            self.opt["training"]["pretrain_path"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(self.opt, resume_state["iter"])  # check resume options

        self.resume_state = True
        
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

        best_psnr = 0.0
        best_iter = 0

        # -------------------------------------------------------------------------
        # ------------------------- training starts here --------------------------
        # -------------------------------------------------------------------------
        for epoch in range(start_epoch, self.total_epochs + 1):
            for idx, batch in enumerate(self.train_loader):
                current_step += 1

                if current_step > self.total_iters:
                    break
                
                total_loss = 0.
                psnr_ls = []
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss, psnr = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        psnr_ls.append(psnr)
                    psnr_mean = np.mean(np.array(psnr_ls))

                    # self.accelerator.backward(loss)
                    total_loss.backward()
                    
                # self.accelerator.wait_for_everyone()
                # self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                


                if current_step % opt["logger"]["print_freq"] == 0:
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
                if current_step % opt["training"]["val_freq"] == 0:  
                    self.val()

                #### save models and training states
                if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                    self.logger.info("Saving models and training states.")
                    self.save(current_step)

        self.logger.info("Saving the final model.")
        self.save("latest")
        self.logger.info("End of Predictor and Corrector training.")
        self.tb_logger.close()

    def val(self):
        avg_psnr = 0.0
        idx = 0
        for it, batch in enumerate(self.val_loader):

            # valid Predictor

            # calculate PSNR
            avg_psnr += util.calculate_psnr(output, gt_img)
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


if __name__ == "__main__":
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="conf/wv3.yaml")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)


    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
