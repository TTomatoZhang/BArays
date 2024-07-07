"""
This script contains functions that are useful to handle satellite images and georeferenced data
"""
import numpy as np
import rasterio
import datetime
import os, sys, re
import shutil
import json
import rpcm
import glob
import cv2
import torch
from torchvision import transforms as T

# from rich.console import Console
# from rich.table import Table
'''
row: y: line
col: x: samp
'''

def GET_PLH_COEF(P, L, H):
    '''
    Args:
        PLH: torch Tensor [B, 3]
    '''
    #P, L, H = torch.hsplit(PLH, 3)
    B = P.shape[0]
    coef = torch.zeros((B, 20)).to(H.device)
    coef[:, 0] = 1.0
    coef[:, 1] = L
    coef[:, 2] = P
    coef[:, 3] = H
    coef[:, 4] = L * P
    coef[:, 5] = L * H
    coef[:, 6] = P * H
    coef[:, 7] = L * L
    coef[:, 8] = P * P
    coef[:, 9] = H * H
    coef[:, 10] = P * L * H
    coef[:, 11] = L * L * L
    coef[:, 12] = L * P * P
    coef[:, 13] = L * H * H
    coef[:, 14] = L * L * P
    coef[:, 15] = P * P * P
    coef[:, 16] = P * H * H
    coef[:, 17] = L * L * H
    coef[:, 18] = P * P * H
    coef[:, 19] = H * H * H

    return coef


def load_gray_tensor_from_rgb_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
        #img.dtype=np.float32()# (3, h, w)  ->  (h, w, 3)
        img_gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
    #h, w = img.shape[:2]
    img_gray = T.ToTensor()(img_gray)
    gray = img_gray.view(1, -1).permute(1, 0)  # (h*w, 3)
    gray = gray.type(torch.FloatTensor)
    return gray

def load_gray_tensor_from_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0))
    #h, w = img.shape[:2]
    img_gray = T.ToTensor()(img)
    gray = img_gray
    gray = gray.type(torch.FloatTensor)
    return gray

def load_gray_tensor_from_geotiff_eq(img_path):
    f = rasterio.open(img_path)
    img = f.read(),
    gray = torch.from_numpy(img[0].astype(np.uint8))
    gray = T.functional.equalize(gray)
    gray = gray.type(torch.FloatTensor)
    return gray

def load_gray_tensor_maxmin_from_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        im = f.read()
        max_h, min_h = im.max(), im.min()
        img = np.transpose(im, (1, 2, 0))
    #h, w = img.shape[:2]
    img_gray = T.ToTensor()(img)
    gray = img_gray
    gray = gray.type(torch.FloatTensor)
    return gray, max_h, min_h

def load_gray_tensor_maxmin_from_geotiff_eq(img_path):
    with rasterio.open(img_path, 'r') as f:
        im = f.read()
        max_h, min_h = im.max(), im.min()
    #h, w = img.shape[:2]
    gray = torch.from_numpy(im.astype(np.uint8))
    gray = T.functional.equalize(gray)
    gray = gray.type(torch.FloatTensor)
    return gray, max_h, min_h

def load_tensor_from_rgb_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    #h, w = h_original // 3, w_original // 3
    img = T.ToTensor()(img)  # (3, h, w)
    #rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = img.permute(1, 2, 0)
    rgbs = rgbs.type(torch.FloatTensor)
    # res = {
    #     'rgbs':rgbs,
    #     'hw': [h, w]
    # }
    return rgbs

def load_tensor_from_rgb_geotiff_eq(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = f.read()

    #h, w = h_original // 3, w_original // 3
    img = torch.from_numpy(img)  # (3, h, w)
    img = T.functional.equalize(img)
    rgbs = img.type(torch.FloatTensor)
    return rgbs

def load_aug_rpc_tensor_from_txt(filepath):
    """
    Read the direct and inverse rpc from a file
    :param filepath:
    :return:
    """
    if os.path.exists(filepath) is False:
        print("Error#001: can't find " + filepath + " in the file system!")
        return

    with open(filepath, 'r') as f:
        all_the_text = f.read().splitlines()

    data = [text.split(' ')[1] for text in all_the_text]
    # print(data)
    # data = np.array(data, dtype=np.float64)
    data = np.array(data, dtype=np.float32)
    data = torch.from_numpy(data)
    return data

def GetH_MAX_MIN(rpc):
    """
    Get the max and min value of height based on rpc
    :return: hmax, hmin
    """
    hmax = rpc[4] + rpc[9]  # HEIGHT_OFF + HEIGHT_SCALE
    hmin = rpc[4] - rpc[9]  # HEIGHT_OFF - HEIGHT_SCALE

    return hmax, hmin

def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}".format(n))
    transformer = Transformer.from_proj(proj_src, proj_dst)
    easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    #transformer = Transformer.from_crs(n, l)
    #easts, norths = transformer.transform(lons, lats)
    #easts, norths =
    return easts, norths

def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc tensor of 170
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    # import copy
    #
    # rpc_scaled = copy.copy(rpc)
    rpc[5] *= float(alpha)  # LINE_SCALE
    rpc[6] *= float(alpha)  # SAMP_SCALE
    rpc[0] *= float(alpha)  # LINE_OFF
    rpc[1] *= float(alpha)  # SAMP_OFF
    # rpc_scaled.LINE_SCALE *= float(alpha)
    # rpc_scaled.SAMP_SCALE *= float(alpha)
    # rpc_scaled.LINE_OFF *= float(alpha)
    # rpc_scaled.SAMP_OFF *= float(alpha)
    return rpc

def save_pfm(file, image, scale=1):
    file = open(file, mode='wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(bytes('PF\n' if color else 'Pf\n', encoding='utf8'))
    file.write(bytes('%d %d\n' % (image.shape[1], image.shape[0]), encoding='utf8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(bytes('%f\n' % scale, encoding='utf8'))

    image_string = image.tostring()
    file.write(image_string)

    file.close()

def load_pfm(fname):
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0)

    return data