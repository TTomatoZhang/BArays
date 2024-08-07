"""
This script defines the dataloader for a dataset of multi-view satellite images
"""


import numpy as np
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import rasterio
import rpcm
import glob
import utils.sat_utils as sat_utils


def get_rays(cols, rows, rpc, min_alt, max_alt):
    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    lons_high, lats_high = rpc.localization(cols, rows, max_alts)
    x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats_high, lons_high, max_alts)
    xyz_near = np.vstack([x_near, y_near, z_near]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    lons_low, lats_low = rpc.localization(cols, rows, min_alts)
    x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats_low, lons_low, min_alts)
    xyz_far = np.vstack([x_far, y_far, z_far]).T

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


class SatelliteDataset(Dataset):
    def __init__(self, root_dir, split="train", downscale=1):
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
        ## json dir
        # json_dir = os.path.join(self.root_dir, scene, 'rpc_ba_crop_json')
        json_dir = os.path.join(self.root_dir, scene, 'rpc_ba_crop_aug')
        
        dsm_dir = os.path.join(self.root_dir, scene, 'dsm')
        dsm_path = glob.glob(dsm_dir + '/*DSM.tif')[0]
        with rasterio.open(dsm_path) as dsmp:
            mat = dsmp.read()
        min_alt, max_alt = mat.min(), mat.max()
        h, w = self.imgH, self.imgW
        ## json glob
        all_json = glob.glob("{}/*.json".format(json_dir))
        
        all_rays = []
        for json_p in all_json:
            ## read rpc dictionary
            d = sat_utils.read_dict_from_json(json_p)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.downscale)
            # d = sat_utils.read_dict_from_txt(json_p)
            # rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d, dict_format="rpcm"), 1.0 / self.downscale)
            
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
            json_name = os.path.basename(json_p)
            cache_path = "{}/{}/{}/{}.data".format(self.root_dir, scene, 'rays_raw', json_name.replace('_PAN.json', ''))
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

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays

    def __len__(self):
        return self.numpairs

    def __getitem__(self, idx):
        rgbpaths = self.rgb_paths[idx]
        rpcpaths = [rgbpath.replace('rgb_crop', 'rpc_ba_crop_json') for rgbpath in rgbpaths]
        rpcrawpaths = [rgbpath.replace('rgb_crop', 'rpc_raw_crop_aug').replace('_PAN', '') for rgbpath in rgbpaths]
        
        images = [sat_utils.load_tensor_from_geotiff(rgbpath, downscale_factor=self.downscale, 
                  return_range=False, normalize=True, imethod=Image.BICUBIC) for rgbpath in rgbpaths]
        

        rpcs_raw = [sat_utils.load_rpcmodel_tensor_from_json(rpcrawpath).unsqueeze(0).repeat(2, 1) for rpcrawpath in rpcrawpaths]
        rpcs = [sat_utils.load_rpcmodel_tensor_from_json(rpcpath) for rpcpath in rpcpaths]
        batch = {}
        batch["rpcs_raw"] = torch.cat(rpcs_raw, 0)
        batch["rpcs"] = torch.cat(rpcs, 0)

        scene = rgbpaths[0].split('/')[-3]
        scenelocpath = os.path.join(self.root_dir, scene, 'scene_1.loc')
        d = sat_utils.read_dict_from_json(os.path.join(scenelocpath))
        center_ = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        range_ = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        image_ls, rays_ls = self.get_masks(images, rays_ls)
        rays_pair = torch.cat(rays_ls, 0)
        rays_pair = self.normalize_rays(rays_pair, center_, range_)
        
        # the init raw rays
        ray_paths = [rgbpath.replace('rgb_crop', 'rays').replace('_RGB.tif', '.data') for rgbpath in rgbpaths]
        rays_ls = [torch.load(raypath).unsqueeze(0) for raypath in ray_paths]
        rays_pair = torch.cat(rays_ls, 0)
        rays_pair = self.normalize_rays(rays_pair, center_, range_)
        
        raw_ray_paths = [rgbpath.replace('rgb_crop', 'rays_raw').replace('_RGB.tif', '.data') for rgbpath in rgbpaths]
        raw_rays_ls = [torch.load(rraypath).unsqueeze(0) for rraypath in raw_ray_paths]
        raw_rays_pair = torch.cat(raw_rays_ls, 0)
        raw_rays_pair = self.normalize_rays(raw_rays_pair, center_, range_)
        
        batch["image"] = torch.stack(image_ls, 0).squeeze()
        batch["rays_raw"] = raw_rays_pair
        batch["rays_gt"] = rays_pair
        return batch

