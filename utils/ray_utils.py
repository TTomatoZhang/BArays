import numpy as np
import torch
import cv2


def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def get_plucker_np(ray_origins, ray_directions, normalize_ray):
    if normalize_ray:
        # Normalize ray directions to unit vectors
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = np.cross(ray_origins, ray_directions, dim=-1)
    # new_ray = np.concatenate([ray_directions, plucker_normal], dim=-1)
    return plucker_normal


def perspective_to_ray(
    R,
    T,
    width,
    height,
    use_half_pix=True,
    use_plucker=True,
    num_patches_x=16,
    num_patches_y=16,
):
    xyd_grid = compute_unit_coordinates(
        width=width,
        height=height,
        use_half_pix=use_half_pix,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )
    w2v = compose_w2v_np(R, T)
    unprojected = unproject_points(
                w2v,
                xyd_grid.reshape(-1, 3), 
            )
        
    unprojected = torch.stack(unprojected, dim=0)  # (N, P, 3)
    # center is the view to world translation
    # P = compose_w2v_np(R, T)
    v2w = np.linalg.inv(w2v)
    origins = v2w[:3, 3]
    origins = origins.repeat(1, num_patches_x * num_patches_y, 1)  # (N, P, 3)
    directions = unprojected - origins

    if use_plucker:
        plucker = get_plucker_np(origins, directions, normalize_ray=True)
    else:
        plucker = None
    return origins, directions, plucker 


def rays_to_eprspective(
    origins,
    directions,
    width,
    height,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
):
    device = origins.device
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)
    R_I = torch.eye(3, device=device)
    T_I = torch.zeros(3, device=device)

    originsI, directionsI, pluckerI  = camera_to_ray_tensor(
    R_I,
    T_I,
    width,
    height,
    use_half_pix=use_half_pix,
    use_plucker=True,
    num_patches_x=num_patches_x,
    num_patches_y=num_patches_y,
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    R = compute_optimal_rotation_alignment(
        I_patch_rays,
        directions,
    )

    # Construct and return rotated camera
    T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    return R, T


def compute_optimal_rotation_alignment_tensor(A, B):
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): (N, 3)
        B (torch.Tensor): (N, 3)

    Returns:
        R (torch.tensor): (3, 3)
    """
    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh


def compute_optimal_rotation_alignment(A, B):
    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = np.linalg.svd(H, full_matrices=True)
    s = np.linalg.det(U @ Vh)
    S_prime = np.diag(np.array([1, 1, np.sign(s)]))
    return U @ S_prime @ Vh


def compute_optimal_rotation_intrinsics(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2
):
    """
    Note: for some reason, f seems to be 1/f.

    Args:
        rays_origin (torch.Tensor): (N, 3)
        rays_target (torch.Tensor): (N, 3)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): (3, 3)
        focal_length (torch.tensor): (2,)
        principal_point (torch.tensor): (2,)
    """
    device = rays_origin.device
    z_mask = torch.logical_and(
        torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold
    )[:, 2]
    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    A, _ = cv2.findHomography(
        rays_origin.cpu().numpy(),
        rays_target.cpu().numpy(),
        cv2.RANSAC,
        reproj_threshold,
    )
    A = torch.from_numpy(A).float().to(device)

    if torch.linalg.det(A) < 0:
        A = -A

    R, L = ql_decomposition(A)
    L = L / L[2][2]

    f = torch.stack((L[0][0], L[1][1]))
    pp = torch.stack((L[2][0], L[2][1]))
    return R, f, pp


def compute_unit_coordinates(
    width, height,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    device=None,
):
    """
    Compute grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to a grid
    where top left corner is (1, 1) and bottom right corner is (-1, -1)).
    """
    dx = width % num_patches_x
    dy = height % num_patches_y
    if use_half_pix:
        min_y = 0
        max_y = height - dy
        min_x = 0
        max_x = width - dx


    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = x / ((width - 1) / 2) - 1
    y_prime = y / ((height - 1) / 2) - 1
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x)], dim=-1)
    return xyd_grid


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    if p_intersect is None:
        return None, None, None, None
    _, p_line_intersect = point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    # I_eps = torch.zeros_like(I_min_cov.sum(dim=-3)) + 1e-10
    # p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # I_min_cov.sum(dim=-3): torch.Size([1, 1, 3, 3])
    # sum_proj: torch.Size([1, 1, 3, 1])

    # p_intersect = np.linalg.lstsq(I_min_cov.sum(dim=-3).numpy(), sum_proj.numpy(), rcond=None)[0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        return None, None
        ipdb.set_trace()
        assert False
    return p_intersect, r


def point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compose_w2v_np(R, T):
    w2v = np.concatenate([R, T], axis=-1).squeeze()
    identity = np.array([0, 0, 0, 1])
    w2v = np.concatenate([w2v, identity], axis=0)
    return w2v


def unproject_points_perspective(
    w2v,
    xy_depth: torch.Tensor,
):
    unprojection_transform = np.linalg.inv(w2v)
    ret = np.matmul(unprojection_transform, xy_depth)
    return ret

def unproject_points_rpc(
    rpc,
    xy_depth: torch.Tensor,
):
    
    
    return ret
