import random
import numpy as np
import torch
import torch.utils.data.distributed

from models.geometry.depth import inv2depth
from models.dbarf_rpc import rayBaRF
from models.render_image import render_single_image
from models.sample_ray import RaySamplerSingleImage
from visualization.cam_visualizer import *
from models.visualization.feature_visualizer import *
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr
from utils.sat_utils import projection
from utils.render_utils import render_rays
from train_ibrnet import IBRNetTrainer
import models.config as config
# torch.autograd.set_detect_anomaly(True)
from rayBaRFTrainer import rayBaRFTrainer



def train(args):
    device = "cuda:{}".format(args.local_rank)

    trainer = rayBaRFTrainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train(args)




