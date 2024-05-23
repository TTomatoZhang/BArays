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
from datasets.utils_pushbroom import *
import cv2
import tqdm
import utils.sat_utils as sat_utils
import rasterio
import glob

def get_rays(cols, rows, rpc, min_alt, max_alt):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """

    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons_high, lats_high = rpc.localization(cols, rows, max_alts)

    x_near, y_near = utm_from_latlon(lats_high, lons_high)
    xyz_near = np.vstack([x_near, y_near, max_alts]).T


    # similarly, the points of minimum altitude are the furthest away from the camera
    lons_low, lats_low = rpc.localization(cols, rows, min_alts)

    x_far, y_far = utm_from_latlon(lats_low, lons_low)
    xyz_far = np.vstack([x_far, y_far, min_alts]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)
    return rays

def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d

def load_rpc_tensor_from_geotiff(img_path, downscale_factor=1, return_model=False):
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
        self.mask_size = 224
        self.split = split

        self.root_dir = root_dir

        assert os.path.exists(self.root_dir), f"root_dir {self.root_dir} does not exist"
        self.downscale = downscale
        self.imgW, self.imgH = 512 / downscale, 512 / downscale

        all_scenes = sorted(os.listdir(self.root_dir))
        self.scenes = all_scenes
        for scene in self.scenes:
            raycachepath = os.path.join(self.root_dir, scene, "rays")
            if not os.path.exists(raycachepath):
            # try:
            #     shutils.rmtree(os.path.join(self.root_dir, scene, "rays"))
            # except:
            #     pass
                self.init_scaling_params(scene)
        self.parsegroups()
        print('finish loading all scene!')

    def get_masks(self, rgb_ls, rays_ls):
        '''
        rgb: 3, h, w
        rays: 7, h, w
        '''
        rgb_ls_, rays_ls_ = [], []
        for rgb, rays in zip(rgb_ls, rays_ls):
            if rgb.ndim == 4:
                rgb = rgb.squeeze()
            rgb_mask1 = rgb[:, :self.mask_size, :self.mask_size]
            rgb_mask2 = rgb[:, 256:256 + self.mask_size, 256:256 + self.mask_size]
            rays = rays.reshape(512, 512, -1)
            rays_mask1 = rays[:self.mask_size, :self.mask_size, :]
            rays_mask2 = rays[256:256 + self.mask_size, 256:256 + self.mask_size, :]
            
            rgb_ls_ += [rgb_mask1.unsqueeze(0), rgb_mask2.unsqueeze(0)]
            rays_ls_ += [rays_mask1.reshape(-1, 8), rays_mask2.reshape(-1, 8)]
        return rgb_ls_, rays_ls_
        

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

    def init_scaling_params(self, scene):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        json_dir = os.path.join(self.root_dir, scene, 'rpc_ba_crop_json')
        dsm_dir = os.path.join(self.root_dir, scene, 'dsm')
        dsm_path = glob.glob(dsm_dir + '/*DSM.tif')[0]
        with rasterio.open(dsm_path) as dsmp:
            mat = dsmp.read()
        min_alt, max_alt = mat.min(), mat.max()
        all_json = glob.glob("{}/*.json".format(json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p)
            h, w = self.imgH, self.imgW
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
            json_name = os.path.basename(json_p)
            cache_path = "{}/{}/{}/{}.data".format(self.root_dir, scene, 'rays', json_name.replace('_PAN.json', ''))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(rays, cache_path)
            print(f"======= finish the ray caching {cache_path}!")
        all_rays = torch.cat(all_rays, 0)
        
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        # os.makedirs(self.cache_dir, exist_ok=True)
        sat_utils.write_dict_to_json(d, f"{self.root_dir}/{scene}/scene_{self.downscale}.loc")
        print("finish the scene location !")

    def normalize_rays(self, rays, center_, range_):
        rays[:, 0] -= center_[0]
        rays[:, 1] -= center_[1]
        rays[:, 2] -= center_[2]
        rays[:, 0] /= range_
        rays[:, 1] /= range_
        rays[:, 2] /= range_
        rays[:, 6] /= range_
        rays[:, 7] /= range_
        return rays

    def __len__(self):
        return self.numpairs

    def __getitem__(self, idx):
        rgbpaths = self.rgb_paths[idx]
        # rpcpaths = [rgbpath.replace('rgb_crop', 'rpc_ba_crop_json').replace('RGB.tif', 'PAN.json') for rgbpath in rgbpaths]
        images = [load_tensor_from_geotiff(rgbpath, downscale_factor=self.downscale, 
                     return_range=False, normalize=True, imethod=Image.BICUBIC) for rgbpath in rgbpaths]
        rpcs_raw = [load_rpc_tensor_from_geotiff(rgbpath).unsqueeze(0).repeat(2, 1) for rgbpath in rgbpaths]
        # rpcs = [load_rpcmodel_tensor_from_json(rpcpath) for rpcpath in rpcpaths]
        batch = {}
        # batch["image"] = torch.stack(images, 0).squeeze()
        # batch["n"] = 3
        # batch["crop_params"] = torch.stack(crop_parameters)
        batch["rpc_raw"] = torch.cat(rpcs_raw, 0)
        # batch["rpc"] = rpcs

        scene = rgbpaths[0].split('/')[-3]
        scenelocpath = os.path.join(self.root_dir, scene, 'scene_1.loc')
        d = sat_utils.read_dict_from_json(os.path.join(scenelocpath))
        center_ = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        range_ = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # ray_path = os.path.join(self.root_dir, scene, 'rays.data')
        ray_paths = [rgbpath.replace('rgb_crop', 'rays').replace('_RGB.tif', '.data') for rgbpath in rgbpaths]
        rays_ls0 = [torch.load(raypath).unsqueeze(0) for raypath in ray_paths]
        
        image_ls, rays_ls = self.get_masks(images, rays_ls0)
        rays_pair = torch.cat(rays_ls, 0)
        rays_pair = self.normalize_rays(rays_pair, center_, range_)
        
        # dsmpath = glob.glob(self.root_dir + scene + '/dsm/*DSM.tif')[0]
        # with rasterio.open(dsmpath) as dsmp:
        #     mat = dsmp.read()
        # min_alt, max_alt = mat.min(), mat.max()
        # for rpcpath in rpcpaths:
        #     with open(rpcpath) as f:
        #         d = json.load(f)
        #     rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
        #     cols, rows = np.meshgrid(np.arange(self.imgW), np.arange(self.imgH))
        #     rays = get_rays(cols, rows, rpc, min_alt, max_alt)
        #     ray_ls += [rays]

        # rays_pair = torch.cat(ray_ls, 0)
        # rays_pair = self.normalize_rays(rays_pair, center_, range_)
        
        batch["image"] = torch.stack(image_ls, 0).squeeze()
        batch["rays_gt"] = rays_pair
        return batch

