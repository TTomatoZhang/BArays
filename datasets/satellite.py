"""
This script defines the dataloader for a datasets of multi-view satellite images
"""

import rpcm
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils_pushbroom import *
import cv2
import tqdm

import rasterio
import glob


def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d


def load_rpc_tensor_from_geotiff(img_path, downscale_factor, return_model):
    if os.path.exists(img_path) is False:
        print("Error#001: cann't find " + img_path + " in the file system!")
        return
    with rasterio.open(img_path) as f:
        rpcmodel = rpcm.RPCModel(f.tags(ns='RPC'), dict_format="geotiff")
    
    if return_model:
        return rpcmodel
    else:
        rpc = rpcmodel.__dict__
        line_off = rpc['row_offset'] / downscale_factor
        samp_off = rpc['col_offset'] / downscale_factor
        line_scale = rpc['row_scale'] / downscale_factor
        samp_scale = rpc['col_scale'] / downscale_factor

        data = [
                line_off,  # 0: line_off
                samp_off,  # 1: samp_off
                rpc['lat_offset'],  # 2: lat_off
                rpc['lon_offset'],  # 3: lon_off
                rpc['alt_offset'],  # 4: alt_off
                line_scale,  # 5: line_scale
                samp_scale,  # 6: samp_scale
                rpc['lat_scale'],  # 7: lat_scale
                rpc['lon_scale'],  # 8: lon_scale
                rpc['alt_scale']  # 9: alt_scale
                ]

        data.extend(rpc['row_num'])   # 10:30
        data.extend(rpc['row_den'])    # 30:50
        data.extend(rpc['col_num'])    # 40:70
        data.extend(rpc['col_den'])    # 70:90


        # print(data)
        data = np.array(data, dtype=np.float64)
        data = torch.from_numpy(data)
        return data


def load_tensor_from_geotiff(img_path, downscale_factor, 
                             return_range=True,
                             normalize=True,
                             imethod=Image.BICUBIC):
    
    with rasterio.open(img_path) as f:
        data = np.transpose(f.read(), (1, 2, 0)).astype(np.float32) 
        h, w = f.height, f.width
        rpc_d = f.rpcs.to_dict()
        
    data = np.transpose(data, (2, 0, 1))
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        # data = np.transpose(data, (2, 0, 1))

        data = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(data))
        # data = np.transpose(data.numpy(), (1, 2, 0))
    elif downscale_factor < 1:
        # upscale
        w = int(w * (1 / downscale_factor))
        h = int(h * (1 / downscale_factor))
        # data = np.transpose(data, (2, 0, 1))
        data = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(data))
    else:
        data = torch.Tensor(data)

    dmax, dmin = data.max(), data.min()
    if normalize:

        data = (data - dmin) / (dmax - dmin)
    mscs = data.unsqueeze(0)    #.view(-1, n_band)  # (h*w, nband)
    if return_range:
        return mscs, dmax, dmin 
    else:
        return mscs


class SatelliteDataset(Dataset):
    def __init__(self, root_dir, split="train", downscale=1):
        self.split = split

        self.root_dir = root_dir

        assert os.path.exists(self.root_dir), f"root_dir {self.root_dir} does not exist"
        self.downscale = downscale

        all_scenes = sorted(os.listdir(self.root_dir))
        self.scenes = all_scenes
            
        self.parsegroups()
        print('finish loading all scene!')


    def parsegroups(self):
        self.rgb_paths = []
        self.alt_ranges = []
        self.names = []
        self.pan_ranges, self.msi_ranges = [], []
        self.numpairs = 0
        for scene in self.scenes:
            dsmpath = glob.glob(os.path.join(self.root_dir, scene, 'dsm/*DSM.tif'))[0]
            with rasterio.open(dsmpath) as dsmf:
                dsm = dsmf.read()
                alt_max, alt_min = dsm.max(), dsm.min()
            
            pairtxt = os.path.join(self.root_dir, scene, 'pairs_3.txt')
            with open(pairtxt) as f:
                alltext = f.read().splitlines()
            if self.split == 'train':
                alltext = alltext[:-4]
            elif self.split == 'test':
                alltext = alltext[-4:-2]
            else:
                alltext = alltext[-2:]
            numpair = len(alltext)
            self.numpairs += numpair
            for i in range(numpair):
                # pair_ls = [int(at) for at in alltext[1:].split(" ")]
                panpaths = alltext[i].split(" ")
                rgbpaths = [panpath.replace('pan_crop', 'rgb_crop').replace('PAN.tif', 'RGB.tif') for panpath in panpaths]
                self.rgb_paths.append(rgbpaths)

                # rpcs = [load_rpc_tensor_from_geotiff(rgbpath, return_model=True) for rgbpath in rgbpaths]
                # names = [os.path.basename(panpath).split('.')[0] for panpath in panpaths]

                # self.rgbs.append(torch.cat(rgbs, dim=0))
                self.alt_ranges.append(torch.Tensor([alt_max, alt_min]))
                # self.rpcs.append(torch.cat(rpcs, dim=0))


            print('finish loading pair paths of ' + scene)


    def __len__(self):
        return self.numpairs


    def __getitem__(self, idx):
        rgbpaths = self.rgb_paths[idx]
        images = [load_tensor_from_geotiff(rgbpath, downscale_factor=self.downscale, 
                     return_range=False, normalize=True, imethod=Image.BICUBIC) for rgbpath in rgbpaths]
        batch = {}
        batch["image"] = torch.stack(images)
        batch["n"] = 3
        # batch["crop_params"] = torch.stack(crop_parameters)

        return batch



