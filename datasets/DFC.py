"""
This script defines the dataloader for a datasets of multi-view satellite images
"""

from torch.utils.data import Dataset
from datasets.utils_pushbroom import *
import glob


#from datasets.rpcm import *
class Dataset_DFC(Dataset):
    def __init__(self, conf, split="train"):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
        """
        self.conf = conf
        self.split = split
        self.scene = str(conf['data.subname'])

        #if split == 'train' or split == 'val':
        self.root_dir = os.path.join(conf['data.root'])
        # else:
        #     self.root_dir = os.path.join(conf['data.root'], 'Test_ex', conf['dataset.subname'])

        assert os.path.exists(self.root_dir), f"root_dir {self.root_dir} does not exist"
        self.impath = os.path.join(self.root_dir, 'image', 'Track3-RGB-crops', self.scene)
        self.tifglob = sorted(glob.glob(self.impath + '/*.tif'))
        self.jsonpath = os.path.join(self.root_dir, 'cams', 'crops_rpcs_raw', self.scene)
        #self.rpcpath = os.path.join(self.root_dir, 'aug_rpc', self.scene)
        #self.rpcglob = glob.glob(self.rpcpath + '/*.txt')
        self.jsonglob = sorted(glob.glob(self.jsonpath + '/*.json'))
        self.rpcglob = sorted(glob.glob(os.path.join(self.root_dir, 'cams', 'crops_rpcs_raw_aug', self.scene, '*.txt')))
        self.heipath = os.path.join(self.root_dir, 'image', 'Track3-Truth',  self.scene+'_DSM.tif')
        numall = len(self.tifglob)
        self.numtest = int(self.conf['data.val_ratio'] * numall)
        self.fetch_all()

    def fetch_all(self):
        self.rpcs = [load_aug_rpc_tensor_from_txt(rpcpath).unsqueeze(0) for rpcpath in self.rpcglob]
        self.imgs = [load_tensor_from_rgb_geotiff(rgbpath).permute(2, 0, 1) for rgbpath in self.tifglob]
        self.Hs, self.Ws = [], []
        self.Heimaxs, self.Heimins = [], []
        self.sun_eles = []
        self.sun_azis = []

        for jsp in self.jsonglob:
            with open(jsp) as f:
                d_ = json.load(f)
            self.Hs.append(d_['height'])
            self.Ws.append(d_['width'])
            self.Heimaxs.append(torch.Tensor([d_['max_alt']]))
            self.Heimins.append(torch.Tensor([d_['min_alt']]))
            # self.Heimaxs.append(d_['max_alt'])
            # self.Heimins.append(d_['min_alt'])
            self.sun_eles.append(d_['sun_elevation'])
            self.sun_azis.append(d_['sun_azimuth'])
        if self.split == 'train':
            self.tifglob = self.tifglob[:-self.numtest]
            self.Hs, self.Ws = self.Hs[:-self.numtest], self.Ws[:-self.numtest]
            self.Heimaxs, self.Heimins = self.Heimaxs[:-self.numtest], self.Heimins[:-self.numtest]
            self.sun_eles, self.sun_azis = self.sun_eles[:-self.numtest], self.sun_azis[:-self.numtest]
            self.imgs, self.rpcs = self.imgs[:-self.numtest], self.rpcs[:-self.numtest]
        else:
            self.tifglob = self.tifglob[-self.numtest:]
            self.Hs, self.Ws = self.Hs[-self.numtest:], self.Ws[-self.numtest:]
            self.Heimaxs, self.Heimins = self.Heimaxs[-self.numtest:], self.Heimins[-self.numtest:]
            self.sun_eles, self.sun_azis = self.sun_eles[-self.numtest:], self.sun_azis[-self.numtest:]
            self.imgs, self.rpcs = self.imgs[-self.numtest:], self.rpcs[-self.numtest:]
        # self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

        self.all = {}
        self.all['rpc'] = self.rpcs  # torch.cat(self.rpcs, dim=0)
        self.all['image'] = self.imgs
        self.all['height'] = self.Hs
        self.all['width'] = self.Ws
        self.all['maxH'] = self.Heimaxs
        self.all['minH'] = self.Heimins
        self.all['sun_eles'] = self.sun_eles
        self.all['sun_azis'] = self.sun_azis
        self.all['idx'] = [i for i in range(len(self.Hs))]

        #for s in self:

        print('data fetching done!')

    def __len__(self):
        # compute length of dataset
        return len(self.tifglob)

    def __getitem__(self, idx):

        sample = {
            "idx": idx,
            "image": self.imgs[idx],
            "rpc": self.rpcs[idx],
            "height": self.Hs[idx],
            "width": self.Ws[idx],
            "sun_elevation": self.sun_azis[idx],
            "sun_azimuth": self.sun_azis[idx],
            "maxH": self.Heimaxs[idx],
            "minH": self.Heimins[idx],
        }

        if self.split != 'train':
            with open(self.jsonglob[idx]) as f:
                d = json.load(f)
            coords = d['geojson']['coordinates']
            center = d['geojson']['center']
            sample.update({
                "coords": coords,
                "center": center,
                "name": os.path.basename(self.tifglob[idx]).split('.')[0]
            })

        return sample

    def setup_loader(self, shuffle=False,drop_last=False):
        loader = torch.utils.data.DataLoader(self,
            batch_size=1,  #self.conf['train.batch_size'] or 1,
            num_workers=self.conf['data.num_workers'],
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False, # spews warnings in PyTorch 1.9 but should be True in general
        )
        print("number of samples: {}".format(len(self)))
        return loader



