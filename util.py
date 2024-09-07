from torch import nn
import torch.nn.functional as F
import torch as t
import numpy as np
import torch
from collections import namedtuple

mse2psnr = lambda x : -10. * t.log(x) / t.log(t.Tensor([10.]))


Transition = namedtuple('Transition', ['state', 'action', 'prd_map','label'])


def work(nerf, observation, robot_T, lock, queue, step, nerf_batch, device, other_device=None):
    observation = torch.from_numpy(observation).to(device) ## now the depth is normalized tp 0~1
    robot_T = torch.from_numpy(robot_T).to(device)
    with torch.no_grad():
        prd_map = nerf.memory_process(observation, robot_T, lock, queue, step, nerf_batch, other_device)
        if other_device==None:
            return prd_map.cpu()
        else:
            return prd_map

def writeSummary(writer,stats,episode_num):
    for key in stats:
        if len(stats[key]) > 0:
            stat_mean = float(np.mean(stats[key]))
            writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=episode_num)
            stats[key] = []
    writer.flush()

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,padding = None):
    if padding == None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes, kernel_size=4):  #0=s(i-1)-2p+k
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )



def adjust_learning_rate(initial_lr,lr_decay_step,episode,optimizer):
    if lr_decay_step > 0:
        learning_rate = 0.9 * initial_lr * (
                lr_decay_step - episode) / lr_decay_step + 0.1 * initial_lr
        if episode > lr_decay_step:
            learning_rate = 0.1 * initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        learning_rate = initial_lr
    return learning_rate

def ssim_loss(x,y):
    x = x.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    y = y.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return t.clamp((1 - ssim) / 2, 0, 1).mean()

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    pred_map = pred_map.unsqueeze(0).unsqueeze(0)
    loss = 0
    weight = 1.
    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
    return loss

def inter_sample(rays_dp,N_sample, half_dist,jitter=True, is_test = False):
    def jitter_fn(z_vals):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = t.cat([mids, z_vals[..., -1:]], -1)
        lower = t.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        d1,d2 = z_vals.shape
        t_rand = t.rand(d1,d2).to(z_vals.device)
        return lower + (upper - lower) * t_rand

    if not hasattr(inter_sample, 'sample'):
        inter_sample.sample = N_sample
        inter_sample.half_dist = half_dist

    if not hasattr(inter_sample, 'z_vals') or not inter_sample.sample == N_sample\
            or not inter_sample.half_dist == half_dist:
        inter_sample.half_dist = half_dist
        inter_sample.sample = N_sample
        far = t.ones_like(rays_dp)*inter_sample.half_dist
        near = t.ones_like(rays_dp)*-1*inter_sample.half_dist
        t_N_vals = t.linspace(0., 1., steps=N_sample).to(rays_dp.device)
        inter_sample.z_vals = near * (1. - t_N_vals) + far * (t_N_vals)


    dp = rays_dp.clone().view(-1)
    tmp = inter_sample.z_vals.clone()
    tmp[dp < half_dist] = tmp[dp < half_dist] - (dp[dp < half_dist].view(-1,1)-half_dist)
    z_vals = tmp + dp.view(-1,1)
    if jitter:
        z_vals = jitter_fn(z_vals)

    return z_vals


def generate_z_vals_and_depths(depth,N_sample = 64,half_dist=1.0,jitter=True):
    rays_dp = t.reshape(depth[0,-1, 0:-1:2,0:-1:2], [-1, 1])*10.
    #rays_dp = t.reshape(depth[0, -1], [-1, 1]) * 10.
    z_vals = inter_sample(rays_dp, N_sample=N_sample, half_dist=half_dist, jitter=jitter)
    return rays_dp, z_vals

def get_rays(H, W, K, c2w):
    if not hasattr(get_rays, 'i') and not hasattr(get_rays, 'j'):
        i, j = t.meshgrid(t.linspace(0, W-1, W), t.linspace(0, H-1, H))
        get_rays.i = i.t()
        get_rays.j = j.t()
    if not hasattr(get_rays, 'dirs'):
        get_rays.dirs = t.stack([(get_rays.i-K[0][2])/K[0][0],-(get_rays.j-K[1][2])/K[1][1],-t.ones_like(get_rays.i)], -1)
    device = c2w.device
    dirs = get_rays.dirs.to(device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = t.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return t.stack([rays_o, rays_d],dim=0) # [2,H,W,3]

def generate_rays_half(poses,H, W, K, z_vals):

    rays = t.stack([get_rays(H, W, K, p) for p in [poses]], 0)  # [1, ro+rd, H, W, 3]

    rays = t.transpose(rays, 1, 2)  # [1, H, ro+rd, W, 3]
    rays = t.transpose(rays, 2, 3)[0,0:-1:2,0:-1:2]  # [1, H, W, ro+rd, 3] = [1,H,W,2,3]
    #rays = t.transpose(rays, 2, 3)
    rays = t.reshape(rays, (-1, 2, 3)).float()  # [H*W, ro+rd, 3] = [H*W,2,3]

    rays_o, rays_d = rays[:, 0], rays[:, 1]    # [H*W, 3]

    viewdirs = rays_d / t.norm(rays_d, dim=-1, keepdim=True)  # 将方向归一化 [H*W, 3]

    pts = rays_o[..., None, :] + viewdirs[..., None, :] * z_vals[..., :, None]  # [N_rays=H*W, N_sample, 3]
    return pts.float(), viewdirs.float()

def minibatch(batch_size, model, pts, viewdirs):
    if len(pts) > batch_size:
        length = len(pts)
        raw, prd = [], []
        for i in range(0, length, batch_size):
            a,b = model(pts[i:i + batch_size], viewdirs[i:i + batch_size])
            raw.append(a)
            prd.append(b)
        return t.cat(raw,0), t.cat(prd,0)
    else:
        raw, prd = model(pts, viewdirs)
        return raw, prd

def render_pred(raw, prd, z_vals,H,W,is_flat=False):
    raw2alpha = lambda raw, dists: 1. - t.exp(-raw * dists)
    device = z_vals.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]   # [N_rays, N_samples]
    dists = t.cat([dists, t.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)

    alpha_raw, rgb_emit, prd_emit = raw[..., 0], raw[..., 1:], prd[..., :]
    alpha = raw2alpha(alpha_raw, dists)
    T = t.cumprod(t.cat([t.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    alpha = alpha * T
    prd_map = t.sum(alpha[..., None] * prd_emit, -2)
    if is_flat:
        return prd_map
    else:
        return prd_map.view(H, W, -1)

def render(raw, z_vals,H=None,W=None,is_flat=False):
    raw2alpha = lambda raw, dists: 1. - t.exp(-raw * dists)
    device = z_vals.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]   # [N_rays, N_samples]
    dists = t.cat([dists, t.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)

    alpha_raw, rgb_emit = raw[..., 0], raw[..., 1:]
    alpha = raw2alpha(alpha_raw, dists)
    T = t.cumprod(t.cat([t.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    alpha = alpha * T
    # prd_map = t.sum(alpha[..., None] * prd_emit, -2)
    rgb_map = t.sum(alpha[..., None] * rgb_emit, -2)
    depth_map = t.sum(alpha * z_vals, -1)
    if is_flat:
        return rgb_map, depth_map
    else:
        assert H != None
        return rgb_map.view(H, W, -1), depth_map.view(H, W)

def loss(rgb_map, depth_map, rgb, depth, is_ssim, is_smooth):
    if is_ssim:
        h,w,_ = rgb_map.shape
        rgb = rgb.view(h,w,-1)
        depth = depth.view(h,w)
        loss_ssim = ssim_loss(rgb_map.view(h,w,-1),rgb)
    else: loss_ssim=0
    if is_smooth:
        loss_smooth = smooth_loss(depth_map)
    else:
        loss_smooth = 0
    loss_rgb_mse = F.mse_loss(rgb_map.float(),rgb)
    loss_depth = F.mse_loss(depth_map.float(),depth)
    loss = 1.*loss_rgb_mse + 1.*loss_depth + 0.05*loss_ssim+0.15*loss_smooth
    # loss = 0.85 * F.mse_loss(depth_map.float(),depth) + 0.15 * loss_ssim + 0.15 * loss_smooth
    return loss, mse2psnr(loss_rgb_mse.cpu())
