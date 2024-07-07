import torch
import torch.nn as nn
import numpy as np
from datasets.utils_pushbroom import *
# for test
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def alphaRenderingRPC(rgb_MPI, sigma_MPI, XYH_coor):
    """
    Rendering image, follow the equation of volume rendering process
    Args:
        rgb_MPI: rgb MPI representation, type:torch.Tensor, shape:[B, Nsample, 3, H, W]
        sigma_MPI: sigma MPI representation, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        XYH_coor: photo2obj coordinates in camera coordinate, shape:[B, Nsample, 3, H, W]

    Returns:
        rgb_syn: synthetic RGB image, type:torch.Tensor, shape:[B, 3, H, W]
        altitude_syn: synthetic height, type:torch.Tensor, shape:[B, 1, H, W]
        transparency_acc: accumulated transparency, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        weights: render weights in per plane and per pixel, type:torch.Tensor, shape:[B, Nsample, 1, H, W]

    """
    B, Nsample, _, H, W = sigma_MPI.shape
    XYH_coor_diff = XYH_coor[:, 1:, :, :, :] - XYH_coor[:, :-1, :, :, :]    # [B, Nsample-1, 1, H, W]
    XYH_coor_diff = torch.norm(XYH_coor_diff, dim=-1, keepdim=True)  # calculate distance, [B, Nsample-1, 3, H, W]
    XYH_coor_diff = torch.cat((XYH_coor_diff,
                               torch.full((B, 1, 1, H, W), fill_value=1e3, dtype=XYH_coor_diff.dtype, device=XYH_coor_diff.device)),
                              dim=1)    # [B, Nsample, H, W, 1]
    transparency = torch.exp(-sigma_MPI * XYH_coor_diff)    # [B, Nsample, 1, H, W]
    alpha = 1 - transparency    # [B, Nsample, 1, H, W]

    alpha_comp_cumprod = torch.cumprod(1 - alpha, dim=1)  # [B, Nsample, 1, H, W]
    preserve_ratio = torch.cat((torch.ones((B, 1, H, W, 1), dtype=alpha.dtype, device=alpha.device),
                                alpha_comp_cumprod[:, 0:Nsample-1, :, :, :]), dim=1)  # [B, Nsample, 1, H, W]
    weights = alpha * preserve_ratio  # [B, Nsample, 1, H, W]
    rgb_syn = torch.sum(weights * rgb_MPI, dim=1, keepdim=False)  # [B, 3, H, W]
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # [B, 1, H, W]
    altitude_syn = torch.sum(weights * XYH_coor[:, :, :, :, 2:], dim=1, keepdim=False) / (weights_sum + 1e-5)  # [B, 1, H, W]

    return rgb_syn, altitude_syn, transparency, weights

def planeVolumeRenderingRPC(rgb_MPI, sigma_MPI, xyz_coor):
    """
    Rendering image, follow the equation of volume rendering process
    Args:
        rgb_MPI: rgb MPI representation, type:torch.Tensor, shape:[B, ndepth, 3, H, W]
        sigma_MPI: sigma MPI representation, type:torch.Tensor, shape:[B, ndepth, 1, H, W]
        xyz_coor: pixel2camera coordinates in camera coordinate, shape:[B, ndepth, 3, H, W]

    Returns:
        rgb_syn: synthetic RGB image, type:torch.Tensor, shape:[B, 3, H, W]
        depth_syn: synthetic depth, type:torch.Tensor, shape:[B, 1, H, W]
        transparency_acc: accumulated transparency, type:torch.Tensor, shape:[B, ndepth, 1, height, width]
        weights: render weights in per plane and per pixel, type:torch.Tensor, shape:[B, ndepth, 1, height, width]

    """
    B, ndepth, _, height, width = sigma_MPI.shape
    xyz_coor_diff = xyz_coor[:, 1:, :, :, :] - xyz_coor[:, :-1, :, :, :]    # [B, ndepth-1, 3, height, width]
    xyz_coor_diff = torch.norm(xyz_coor_diff, dim=2, keepdim=True)  # calculate distance, [B, ndepth-1, 1, height, width]
    xyz_coor_diff = torch.cat((xyz_coor_diff,
                               torch.full((B, 1, 1, height, width), fill_value=1e3, dtype=xyz_coor_diff.dtype, device=xyz_coor_diff.device)),
                              dim=1)    # [B, ndepth, 1, height, width]
    transparency = torch.exp(-sigma_MPI * xyz_coor_diff)    # [B, ndepth, 1, height, width]
    alpha = 1 - transparency    # [B, ndepth, 1, height, width]

    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)    # [B, ndepth, 1, height, width]
    transparency_acc = torch.cat((torch.ones((B, 1, 1, height, width), dtype=transparency_acc.dtype, device=transparency_acc.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1) # [B, ndepth, 1, height, width]

    weights = transparency_acc * alpha  # [B, ndepth, 1, height, width]
    h = transparency * sigma_MPI
    # calculate rgb_syn, depth_syn
    rgb_syn = torch.sum(weights * rgb_MPI, dim=1, keepdim=False)    # [B, 3, height, width]
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # [B, 1, height, width]
    depth_syn = torch.sum(weights * xyz_coor[:, :, 2:, :, :], dim=1, keepdim=False) / (weights_sum + 1e-5)  # [B, 1, height, width]
    return rgb_syn, depth_syn, transparency_acc, weights


def sampleAltitude(altitude_min, altitude_max, altitude_hypothesis_num):
    """
    Uniformly sample altitude from [inversed_altitude_max, inversed_altitude_max]
    Args:
        altitude_min: min altitude value, type:torch.Tensor, shape:[B,]
        altitude_max: max altitude value, type:torch.Tensor, shape:[B,]
        altitude_hypothesis_num: altitude hypothesis number, type:int

    Returns:
        altitude_sample: altitude sample, type:torch.Tensor, shape:[B, Nsample]
    """
    altitude_samples = []
    for i in range(altitude_min.shape[0]):
        altitude_samples.append(torch.linspace(start=altitude_min[i].item(), end=altitude_max[i].item(), steps=altitude_hypothesis_num, device=altitude_min.device))
    altitude_sample = torch.stack(altitude_samples, dim=0)    # [B, Nsample]
    return altitude_sample.flip(-1)



def GET_PLH_COEF(P, L, H):
    """
    :param PLH: standardized 3D position XYH for src_RPC, type:torch.Tensor, shape:[B * Nsample, 1, H * W]
    :return: coef: type:torch.Tensor, shape:[B * Nsample * H * W, 20]
    """
    device = P.device
    P, L, H = P.squeeze(), L.squeeze(), H.squeeze()
    n_num = P.shape[0]
    #coef = np.zeros((n_num, 20))
    with torch.no_grad():
        coef = torch.ones((n_num, 20), dtype=torch.double).to(device)
        # coef[:, 0] = 1.0
        #try:
        coef[:, 1] = L
        # except:
        #     print('a')
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
        # coef[:, 1] = L
        # coef[:, 2] = P
        # coef[:, 3] = H
        # coef[:, 4] = L * P
        # coef[:, 5] = L * H
        # coef[:, 6] = P * H
        # coef[:, 7] = L * L
        # coef[:, 8] = P * P
        # coef[:, 9] = H * H
        # coef[:, 10] = P * L * H
        # coef[:, 11] = L * L * L
        # coef[:, 12] = L * P * P
        # coef[:, 13] = L * H * H
        # coef[:, 14] = L * L * P
        # coef[:, 15] = P * P * P
        # coef[:, 16] = P * H * H
        # coef[:, 17] = L * L * H
        # coef[:, 18] = P * P * H
        # coef[:, 19] = H * H * H

    return coef

def PROJECTION(rpc, Lat, Lon, H):
    """
    From (lat: X, lon: Y, hei: Z) to (samp, line) using the direct rpc
    Args:
        rpc: tgt RPC
        XYH: 3D position for src_RPC, type:torch.Tensor, shape:[B, Nsample, 3,  H,  W]
    Returns:
        samplineheight_grid: shape:[B, Nsample, 3,  H, W]
    """
    device = rpc.device
    # B, Nsample, _, H_im, W_im = XYH.shape
    # XYH = XYH.permute(0, 1, 3, 4, 2)
    # XYH = XYH.reshape(-1, 3)
    # lat, lon, H = torch.hsplit(XYH, 3) # torch.split(XYH, [1,1,1], -1)


    with torch.no_grad():
        lat = Lat.clone()
        lon = Lon.clone()
        hei = H.clone()

        # lat -= rpc[2].view(-1, 1) # self.LAT_OFF
        # lat /= rpc[7].view(-1, 1) # self.LAT_SCALE
        #
        # lon -= rpc[3].view(-1, 1) # self.LONG_OFF
        # lon /= rpc[8].view(-1, 1) # self.LONG_SCALE
        #
        # hei -= rpc[4].view(-1, 1) # self.HEIGHT_OFF
        # hei /= rpc[9].view(-1, 1) # self.HEIGHT_SCALE

        lat -= rpc[2]  # self.LAT_OFF
        lat /= rpc[7]  # self.LAT_SCALE

        lon -= rpc[3]  # self.LONG_OFF
        lon /= rpc[8]  # self.LONG_SCALE

        hei -= rpc[4]  # self.HEIGHT_OFF
        hei /= rpc[9]  # self.HEIGHT_SCALE

        coef = GET_PLH_COEF(lat, lon, hei)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = torch.sum(coef * rpc[50: 70].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[70:90].view(-1, 1, 20), dim=-1)
        line = torch.sum(coef * rpc[10: 30].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[30:50].view(-1, 1, 20), dim=-1)

        samp *= rpc[6]  # self.SAMP_SCALE
        samp += rpc[1]  # self.SAMP_OFF

        line *= rpc[5]  # self.LINE_SCALE
        line += rpc[0]  # self.LINE_OFF

        samp = samp.permute(1, 0)
        line = line.permute(1, 0)
    return samp, line  # col, row

def LOCALIZATION(rpc, S, L, H):
    """
    From (samp: S, line: L, hei: H) to (lat, lon) using the inverse rpc
    photo to object space
    Args:
        rpc: src RPC
        SLH: 3D position for src_RPC, type:torch.Tensor, shape: [B, Nsample, 3, H, W]
    Returns:
        XYH: 3D position for src_RPC in object space, type:torch.Tensor, shape: [B, Nsample, 3, H, W]
    """
    device = rpc.device
    with torch.no_grad():
        # torch.cuda.synchronize()
        # t0 = time.time()
        samp = S.clone()
        line = L.clone()
        hei = H.clone()

        samp -= rpc[1].view(-1, 1)  # self.SAMP_OFF

        samp /= rpc[6].view(-1, 1)  # self.SAMP_SCALE

        line -= rpc[0].view(-1, 1)  # self.LINE_OFF
        line /= rpc[5].view(-1, 1)  # self.LINE_SCALE

        hei -= rpc[4].view(-1, 1)  # self.HEIGHT_OFF
        hei /= rpc[9].view(-1, 1)  # self.HEIGHT_SCALE
        # t1 = time.time()
        coef = GET_PLH_COEF(samp, line, hei)
        # torch.cuda.synchronize()
        # t2 = time.time()

        # coef: (B, ndepth*H*W, 20) rpc[:, 90:110] (B, 20)
        lat = torch.sum(coef * rpc[90:110].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[110:130].view(-1, 1, 20), dim=-1)
        lon = torch.sum(coef * rpc[130:150].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[150:170].view(-1, 1, 20), dim=-1)

        # torch.cuda.synchronize()
        # t3 = time.time()
        lat *= rpc[7].view(-1, 1)
        lat += rpc[2].view(-1, 1)

        lon *= rpc[8].view(-1, 1)
        lon += rpc[3].view(-1, 1)
        lat = lat.permute(1, 0)
        lon = lon.permute(1, 0)
    return lat, lon

def project_src2tgt(src_rgb_syn, src_alt_syn, src_rpc, tgt_rpc):
    B, dim, H, W = src_rgb_syn.shape
    device = src_rpc.device
    #assert B == 1, "B must be 1"
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),  # row: line
                               torch.arange(0, W, dtype=torch.double, device=device)])  # col: samp
        yv, xv = y.contiguous(), x.contiguous()  # [H, W, 1]
        xv = xv.view(-1, 1)  # [H * W, 1]
        yv = yv.view(-1, 1)  # [H * W, 1]
        h = src_alt_syn.squeeze().view(H*W, 1)  # [H, W, 1] penalize on the depth
        h = h.double()

        lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
        samp, line = PROJECTION(tgt_rpc, lat, lon, h)  # [1, H*W]

        samp = samp.float().permute(1,0)
        line = line.float().permute(1,0)

        tgt_x_normalized = samp / ((W - 1) / 2) - 1  # [1, H * W, 1]
        tgt_y_normalized = line / ((H - 1) / 2) - 1  # [1, H * W, 1]
        #tgt_x, tgt_y = tgt_x_normalized.view(H, W, 1), tgt_y_normalized.view(H, W, 1)
        tgt_grid = torch.stack((tgt_x_normalized, tgt_y_normalized), dim=-1)#.view(B, H, W, 2)   # [H * W, 2]
        # tgt_sampled_rgb = tgt_rgb_syn[:, tgt_y, tgt_x]  # [1, H, W]
    projected_tgt = torch.nn.functional.grid_sample(
        #tgt_rgb_syn,  # img.unsqueeze(0) # [B, C, H, W]
        src_rgb_syn.to(torch.float32),
        tgt_grid.unsqueeze(2),    # [1, H * W, 1, 2]
        align_corners=False,
        mode='bilinear', padding_mode='zeros').squeeze()
    projected_tgt = projected_tgt.view(dim, H, W).unsqueeze(0)  # [B, C, H, W]
    # mask
    tgt_grif = tgt_grid.view(H, W, 2)
    valid_mask_x = torch.logical_and(tgt_grid[:, :, 0] < W, tgt_grid[:, :, 0] > -1)  # [H * W, 2]
    valid_mask_y = torch.logical_and(tgt_grid[:, :, 1] < H, tgt_grid[:, :, 1] > -1)
    valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # [B*Nsample, H_render, W_render]
    valid_mask = valid_mask.reshape(B, 1, H, W)  # [B, Nsample, H_render, W_render]

    return projected_tgt, valid_mask

def render_rays(model,
                rays,
                feat_maps,
                N_samples,
                N_importance=0,
                depth_prior=None,
                rel_rays=None
                ):
    use_disp = False
    perturb = 1.0

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]

    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid pointsloss
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    # discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)



    return result




    