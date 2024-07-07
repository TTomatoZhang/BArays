
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')


class DeepVoxelsDataset(Dataset):
    def __init__(self, args, subset,
                 scenes='vase',  # string or list
                 **kwargs):

        self.folder_path = os.path.join(args.rootdir, '/root/ztt/satdata/satmvsdata/JAX_004')
        self.subset = subset  # train / test / validation
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        if isinstance(scenes, str):
            scenes = [scenes]

        self.scenes = scenes
        self.all_rgb_files = []
        self.all_depth_files = []
        self.all_rpc_files = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, subset, scene)

            rgb_files = [os.path.join(self.scene_path, 'rgb', f)
                         for f in sorted(os.listdir(os.path.join(self.scene_path, 'rgb')))]
            if self.subset != 'train':
                rgb_files = rgb_files[::self.testskip]
            depth_files = [f.replace('rgb', 'depth') for f in rgb_files]
            rpc_files = [f.replace('rgb', 'cams').replace('tif', 'json') for f in rgb_files]

            self.all_rgb_files.extend(rgb_files)
            self.all_depth_files.extend(depth_files)
            self.all_rpc_files.extend(rpc_files)


    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.all_rgb_files)
        rgb_file = self.all_rgb_files[idx]
        rpc_file = self.all_rpc_files[idx]

        train_rgb_files = sorted(glob.glob(os.path.join(self.scene_path.replace('/{}/'.format(self.subset),
                                                                                '/train/'), 'rgb', '*')))
        train_rpcs_files = [f.replace('rgb', 'cams').replace('tif', 'json') for f in train_rgb_files]
        train_rpcs = np.stack([np.loadtxt(file).reshape(4, 4) for file in train_rpcs_files], axis=0)

        if self.subset == 'train':
            id_render = train_rpcs_files.index(rpc_file)
            subsample_factor = np.random.choice(np.arange(1, 5))
            num_source_views = np.random.randint(low=self.num_source_views-4, high=self.num_source_views+2)
        else:
            id_render = -1
            subsample_factor = 1
            num_source_views = self.num_source_views

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_rpc = np.loadtxt(rpc_file).reshape(4, 4)

        img_size = rgb.shape[:2]

        nearest_rpc_ids = get_nearest_rpc_ids(render_rpc,
                                                train_rpcs,
                                                min(num_source_views*subsample_factor, 40),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_rpc_ids = np.random.choice(nearest_rpc_ids, num_source_views, replace=False)

        assert id_render not in nearest_rpc_ids
        # occasionally include target image in the source views
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.subset == 'train':
            nearest_rpc_ids[np.random.choice(len(nearest_rpc_ids))] = id_render

        src_rgbs = []
        src_rpcs = []
        for id in nearest_rpc_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_rpc = train_rpcs[id]
            if self.rectify_inplane_rotation:
                src_rpc, src_rgb = rectify_inplane_rotation(train_rpc, render_rpc, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                         train_rpc.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        origin_depth = np.linalg.inv(render_rpc.reshape(4, 4))[2, 3]

        if 'cube' in rgb_file:
            near_depth = origin_depth - 1.
            far_depth = origin_depth + 1
        else:
            near_depth = origin_depth - 0.8
            far_depth = origin_depth + 0.8

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'scene_path': self.scene_path
                }

