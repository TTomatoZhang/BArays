"""
This script contains functions that are useful to handle satellite images and georeferenced data
"""

import numpy as np
import rasterio
import datetime
import os
import shutil
import json
import glob
import rpcm
import torch
import pyproj
from typing import NamedTuple
from PIL import Image
from torchvision import transforms as T
from models.base.functools import lru_cache

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


def load_rpcmodel_tensor_from_json(filepath):
    """
    Read the direct and inverse rpc from a file
    :param filepath:
    :return:
    """
    if os.path.exists(filepath) is False:
        print("Error#001: can't find " + filepath + " in the file system!")
        return
    with open(filepath) as f:
        d = json.load(f)
    d_rpc = d["rpc"]
    rpc_m = rpcm.RPCModel(d_rpc, dict_format="geotiff")
    return rpc_m


def get_tensor_from_rpc(rpc):
    '''
           LINE_OFF, SAMP_OFF, LAT_OFF, LONG_OFF, HEIGHT_OFF,
           LINE_SCALE, SAMP_SCALE, LAT_SCALE, LONG_SCALE, HEIGHT_SCALE,
           'LINE_NUM_COEFF,
           LINE_DEN_COEFF,
           SAMP_NUM_COEFF,
           SAMP_DEN_COEFF
    '''
    data = [rpc.row_offset, rpc.col_offset, rpc.lat_offset, rpc.lon_offset, rpc.alt_offset,
            rpc.row_scale, rpc.col_scale, rpc.lat_scale, rpc.lon_scale, rpc.alt_scale]

    data.extend(rpc.row_num)
    data.extend(rpc.row_den)
    data.extend(rpc.col_num)
    data.extend(rpc.col_den)

    return torch.Tensor(data).unsqueeze(0)


def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]


def read_dict_from_txt(input_path):
    with open(input_path) as f:
        d = json.loads(f.read())
    
    # with open(input_path) as f:
    #     d = json.load(f)    
    return d


def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d


def write_dict_to_json(d, output_path):
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    return d


def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    vec = np.array(v).ravel()
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset


def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled

def rescale_rpc_batch(rpcs, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy
    rpcs_scaled = []
    for rpc in rpcs:
        rpc_scaled = copy.copy(rpc)
        rpc_scaled.row_scale *= float(alpha)
        rpc_scaled.col_scale *= float(alpha)
        rpc_scaled.row_offset *= float(alpha)
        rpc_scaled.col_offset *= float(alpha)
        rpcs_scaled.append(rpc)
    return rpc_scaled

def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt

def ecef_to_latlon_torch(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = torch.sqrt((x ** 2) + (y ** 2))
    th = torch.arctan2(a * z, b * p)
    lon = torch.arctan2(y, x)
    lat = torch.arctan2((z + (ep ** 2) * b * (torch.sin(th) ** 3)), (p - esq * a * (torch.cos(th) ** 3)))
    N = a / (torch.sqrt(1 - esq * (torch.sin(lat) ** 2)))
    alt = p / torch.cos(lat) - N
    lon = lon * 180 / torch.pi
    lat = lat * 180 / torch.pi
    return lat, lon, alt

def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    # n = utm.latlon_to_zone_number(lats[0], lons[0])
    # l = utm.latitude_to_zone_letter(lats[0])
    # proj_src = pyproj.Proj("+proj=latlong")
    # proj_dst = pyproj.Proj("+proj=utm +zone={}{}".format(n, l))
    n = utm.latlon_to_zone_number(lats[0], lons[0])
    # n = utm.conversion.zone_number_to_central_longitude(n)
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}".format(n))
    transformer = Transformer.from_proj(proj_src, proj_dst)
    easts, norths = transformer.transform(lons, lats)
    #easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    return easts, norths

def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    from osgeo import gdal

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "/home/aipt/Documents/ztt/satensorf_output/tmp/tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "/home/aipt/Documents/ztt/satensorf_output/tmp/tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(in_dsm_path)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    # ds = None
    # os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, source_path, crop_path))
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    # try:
    # import dsmr
    # except:
    #     print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
    #     fix_xy = True
    # if fix_xy:
    pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
    with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
        dst.write(pred_rdsm, 1)
    # else:
    # import dsmr
    # transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
    # dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
    # with rasterio.open(pred_rdsm_path, "r") as f:
    #     pred_rdsm = f.read()[0, :, :]
    err = pred_rdsm - gt_dsm


    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutils.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)

    return err

def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, gt_dir, out_dir, epoch_number, dsm):
    # save = True
    # save dsm errs
    aoi_id = src_id[:7]
    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
    gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))
    assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"
    from utils.sat_utils import dsm_pointwise_diff
    gt_roi_metadata = np.loadtxt(gt_roi_path)
    rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff_epoch{}.tif".format(src_id, epoch_number))
    rdsm_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                                       out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)
    #os.system(f"rm tmp*.tif.xml")
    # if not save:
    #     os.remove(rdsm_diff_path)
    #     os.remove(rdsm_path)
    return np.nanmean(abs(diff.ravel()))

def dsm_mae(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None):
    abs_err = dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=gt_mask_path)
    return np.nanmean(abs_err.ravel())


def RPC_Obj2Photo(lat, lon, hei, rpc):
    # inlat:  (H * W)
    # inlon:  (H * W)
    # inhei:  (H * W)
    # rpc: (170)

    with torch.no_grad():

        lat -= rpc[2] # self.LAT_OFF
        lat /= rpc[7] # self.LAT_SCALE

        lon -= rpc[3] # self.LONG_OFF
        lon /= rpc[8] # self.LONG_SCALE

        hei -= rpc[4] # self.HEIGHT_OFF
        hei /= rpc[9] # self.HEIGHT_SCALE

        coef = RPC_PLH_COEF(lat, lon, hei)

        rpc = rpc.to(coef.device)
        samp = torch.sum(coef * rpc[50:70].view(1, 20), dim=-1) / torch.sum(coef * rpc[70:90].view(1, 20), dim=-1)
        line = torch.sum(coef * rpc[10:30].view(1, 20), dim=-1) / torch.sum(coef * rpc[30:50].view(1, 20), dim=-1)

        samp *= rpc[6] # self.SAMP_SCALE
        samp += rpc[1] # self.SAMP_OFF

        line *= rpc[5] # self.LINE_SCALE
        line += rpc[0] # self.LINE_OFF
        a = rpcm.RPCModel

    return samp, line


def RPC_PLH_COEF(P, L, H):
    # P, L, H: (H * W)
    # coef:(H * W, 20)
    coef = torch.zeros(P.shape[0], 20).to(P.device)
    with torch.no_grad():
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * coef[:, 5]
        coef[:, 11] = L * coef[:, 7]
        coef[:, 12] = L * coef[:, 8]
        coef[:, 13] = L * coef[:, 9]
        coef[:, 14] = L * coef[:, 4]
        coef[:, 15] = P * coef[:, 8]
        coef[:, 16] = P * coef[:, 9]
        coef[:, 17] = L * coef[:, 5]
        coef[:, 18] = P * coef[:, 6]
        coef[:, 19] = H * coef[:, 9]
    return coef


def apply_poly(poly, x, y, z):
    """
    Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.

    Args:
        poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
            ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the polynom on the input point(s).
    """
    x, y, z = x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)
    out = torch.zeros(x.shape).to(x.device)
    out += poly[0]
    out += poly[1]*y+ poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out


def apply_rfm(num, den, x, y, z):
    """
    Evaluates a Rational Function Model (rfm), on a triplet of numbers.

    Args:
        num: list of the 20 coefficients of the numerator
        den: list of the 20 coefficients of the denominator
            All these coefficients are ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the rfm on the input point(s).
    """
    return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)


def projection(lon, lat, alt, rpc):
    """
    data = [rpc.row_offset, rpc.col_offset, rpc.lat_offset, rpc.lon_offset, rpc.alt_offset, rpc.row_scale, rpc.col_scale,
            rpc.lat_scale, rpc.lon_scale, rpc.alt_scale]

    data.extend(rpc.row_num) [10:30]
    data.extend(rpc.row_den) [30, 50]
    data.extend(rpc.col_num) [50:70]
    data.extend(rpc.col_den)
    """
    nlon = (lon - rpc[3]) / rpc[8]
    nlat = (lat - rpc[2]) / rpc[7]
    nalt = (alt - rpc[4]) / rpc[9]
    # nlon = (np.asarray(lon) - self.lon_offset) / self.lon_scale
    # nlat = (np.asarray(lat) - self.lat_offset) / self.lat_scale
    # nalt = (np.asarray(alt) - self.alt_offset) / self.alt_scale

    col = apply_rfm(rpc[50:70], rpc[70:90], nlat, nlon, nalt)
    row = apply_rfm(rpc[10:30], rpc[30:50], nlat, nlon, nalt)

    # col = apply_rfm(self.col_num, self.col_den, nlat, nlon, nalt)
    # row = apply_rfm(self.row_num, self.row_den, nlat, nlon, nalt)


    col = col * rpc[6] + rpc[1]
    row = row * rpc[5] + rpc[0]
    # col = col * self.col_scale + self.col_offset
    # row = row * self.row_scale + self.row_offset

    return col, row


def load_tensor_from_rgb_geotiff(img_path, downscale_factor, imethod=Image.BICUBIC):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs


def init_scaling_params(rgb_dir, out_dir, img_downscale=1):
    print("Could not find a scene.loc file in the root directory, creating one...")
    print("Warning: this can take some minutes")
    all_rgb = glob.glob("{}/*.tif".format(rgb_dir))
    numall = len(all_rgb)
    dsm_p = glob.glob(rgb_dir.replace('rgb_crop_', 'dsm') + '/*.tif')[0]
    with rasterio.open(dsm_p) as dsm_f:
        dsm = dsm_f.read()
    min_alt, max_alt = dsm.min(), dsm.max()
    all_rays = []
    for rgb_p in all_rgb:
        f = rasterio.open(rgb_p)
        h, w = f.shape
        h, w = int(h // img_downscale), int(w // img_downscale)
        rpc = rescale_rpc(rpcm.RPCModel(f.tags(ns='RPC'), dict_format="geotiff"), 1.0 / img_downscale)
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
    # np.savez(out_dir + '/points.npz', near_points=near_points, far_points=far_points)
    
    print("... finish initing scale params !")
    return d


def init_points(rgb_dir, cache_dir,  N_samples=32, img_downscale=1):
    print("Could not find a scene.loc file in the root directory, creating one...")
    print("Warning: this can take some minutes")
    numall = len(all_rgb)
    all_rgb = glob.glob("{}/*.tif".format(rgb_dir))
    dsm_p = glob.glob(rgb_dir.replace('rgb_crop_', 'dsm') + '/*.tif')[0]
    with rasterio.open(dsm_p) as dsm_f:
        dsm = dsm_f.read()
    min_alt, max_alt = dsm.min(), dsm.max()
    all_rays = []
    for rgb_p in all_rgb:
        f = rasterio.open(rgb_p)
        h, w = f.shape
        h, w = int(h // img_downscale), int(w // img_downscale)
        rpc = rescale_rpc(rpcm.RPCModel(f.rpcs, dict_format="geotiff"), 1.0 / img_downscale)
        cols, rows = np.meshgrid(np.arange(0, w, numall), np.arange(0, h, numall))
        rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
        all_rays += [rays]
    all_rays = np.concatenate(all_rays, axis=0)
    # all_rays[:, :3] -= center_
    # rays[:, 0] /= range_
    # rays[:, 1] /= range_
    # rays[:, 2] /= range_
    # rays[:, 6] /= range_
    # rays[:, 7] /= range_  
    
    d = read_dict_from_json(os.path.join(cache_dir, "scene.loc"))
    center_ = np.array([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
    range_ = np.max(np.array([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))
    # near_points = all_rays[:, :3]
    # far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
    rays_o, rays_d, near, far = all_rays[:, 0:3], all_rays[:, 3:6], all_rays[:, 6:7], all_rays[:, 7:8]
    rays_o -= center_
    rays_o /= range_
    near /= range_
    far /= range_
    
    z_steps = np.linspace(0, 1, N_samples)
    z_vals = near * (1-z_steps) + far * z_steps

    # perturb = 1.0
    # if perturb > 0:  # perturb sampling depths (z_vals)
    #     z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid pointsloss
    #     # get intervals between samples
    #     upper = np.concatenate([z_vals_mid, z_vals[:, -1:]], -1)
    #     lower = np.concatenate([z_vals[:, :1], z_vals_mid], -1)

    #     perturb_rand = perturb * np.random.randn(*z_vals.shape)
    #     z_vals = lower + (upper - lower) * perturb_rand

    #     discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
 
    points_dir = os.path.join(cache_dir, "points.npy")
    np.save(points_dir, xyz)
    print("... finish caching the points edges !")

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = custom_meshgrid([ys, xs]) # torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid

def sample_points_from_depth(self, depth):  
    if depth.ndim == 3:
        C, H, W = depth.shape
        B = 1
    elif depth.ndim == 4:
        B, C, H, W = depth.shape
    else:
        H, W = depth.shape
        B = 1
        C = 1
    # print(f'depth shape: {depth.shape}')
    assert C == 1        
    # Create flat index grid
    grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
    Xw = grid * depth # [B, 3, H, W] * [B, 1, H, W]

    Xw = Xw.reshape(B, 3, -1)

    return Xw

def GetXYHfromAltitudes(H, W, alt_sample, rpc):
    """
    Generate src-view planes 3D position XYH
    Args:
        H: rendered image height, type:int
        W: rendered image width, type:int
        alt_sample: altitude sample in src-view, corresponding to MPI planes' Z, type:torch.Tensor, shape:[B, Nsample]
        rpc: [170,]

    Returns:
        XYH_src: 3D position in src-RPC, type:torch.Tensor, shape:[B, Nsample, 3, H, W]

    """
    device = rpc.device
    # generate meshgrid for src-view.
    B, Nsample = alt_sample.shape
    alt_sample = alt_sample.view(B * Nsample).to(device)

    y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),
                           torch.arange(0, W, dtype=torch.double, device=device)])
    y, x = y.contiguous(), x.contiguous()
    z = torch.ones_like(x)  # cpu

    meshgrid_flat = torch.stack((x, y, z), dim=2).to(device)  # [H, W, 3]

    meshgrid = meshgrid_flat.unsqueeze(0).repeat(B*Nsample, 1, 1, 1).reshape(B*Nsample, H, W, 3).permute(1, 2, 3, 0)  # [H, W, 3, 32]
    meshgrid[:, :, 2, :] = meshgrid[:, :, 2, :] * alt_sample

    #meshgrid = meshgrid.contiguous().view(H*W*B*Nsample, 3)

    ''' 
    meshgrid0 = meshgrid.permute(3, 0, 1, 2)
    meshgrid1 = meshgrid0[5:10, :, :, :]
    meshgrid2 = meshgrid1.contiguous().view(-1, 3)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    xyh = meshgrid2.detach().cpu().numpy()
    # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
    fig.show()

    '''
    # line off [0], samp off [1], line scale[5], samp off [6]
    # todo: test whether need normalization
    # meshgrid[:, 0] -= rpc[1]  # sampoff
    # meshgrid[:, 1] -= rpc[0]  # lineoff
    # meshgrid[:, 0] /= rpc[6]
    # meshgrid[:, 1] /= rpc[5]

    # expand to Nsamples of depth level # [H, W, 3, 32]
    XYH_src = meshgrid.permute(3, 2, 0, 1).contiguous().view(B, Nsample, 3, H, W)  # [B, Nsample, 3, H, W] # right
                                                                  #[1, 32, 3, 262144], [1, 32, 1, 1]
    # for plot
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    ndepth = XYH_src.shape[1]
    XYZ = XYH_src.squeeze().view(ndepth, 3, -1).permute(0, 2, 1).detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(ndepth):
        xyh = XYZ[i, :, :]
        ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
    # xyh = meshgrid.reshape(-1, 3).detach().cpu().numpy()
    fig.show()
    '''
    return XYH_src
