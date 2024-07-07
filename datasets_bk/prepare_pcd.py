import rasterio
import cv2
import numpy as np
import glob
from utils.sat_utils import *
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def init_scaling_params(json_dir, out_dir, img_downscale):
    print("Could not find a scene.loc file in the root directory, creating one...")
    print("Warning: this can take some minutes")
    all_json = glob.glob("{}/*.json".format(json_dir))
    all_rays = []
    for json_p in all_json:
        d = read_dict_from_json(json_p)
        h, w = int(d["height"] // img_downscale), int(d["width"] // img_downscale)
        rpc = rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / img_downscale)
        min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
        all_rays += [rays]
    all_rays = np.concatenate(all_rays, axis=0)
    near_points = all_rays[:, :3]
    far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
    all_points = np.concatenate([near_points, far_points], axis=0)

    d = {}
    d["X_scale"], d["X_offset"] = rpc_scaling_params(all_points[:, 0])
    d["Y_scale"], d["Y_offset"] = rpc_scaling_params(all_points[:, 1])
    d["Z_scale"], d["Z_offset"] = rpc_scaling_params(all_points[:, 2])
    write_dict_to_json(d, f"{out_dir}/scene_{img_downscale}.loc")
    
    print("... done !")

def prepare_random_pcs(loc_dir):
    d = read_dict_from_json(os.path.join(loc_dir, "scene.loc"))
    center_ = np.array([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
    range_ = torch.max(np.array([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))
    rays = 
    rays[:, :3] -= center_
    rays[:, 0] /= range_
    rays[:, 1] /= range_
    rays[:, 2] /= range_
    rays[:, 6] /= range_
    rays[:, 7] /= range_


def get_depth_from_dust3r(input_path, dust3r_path):
    with rasterio.open(input_path) as f:
        mat = f.read()
        mat = (mat - mat.min()) / (mat.max() - mat.min())
    