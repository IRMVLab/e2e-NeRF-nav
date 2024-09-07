import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util import *
from torch.optim import Adam,SGD
from random import choice,sample
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from util import writeSummary
from torchvision.models import resnet18
from copy import deepcopy
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass

def Quat2Rotation(x,y,z,w):
    l1 = torch.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],dim=0)
    l2 = torch.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],dim=0)
    l3 = torch.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2], dim=0)
    T_w = torch.stack([l1,l2,l3],dim=0)
    return T_w
def Rotation2Quat(pose):
    m11,m22,m33 = pose[0][0],pose[1][1],pose[2][2]
    m12,m13,m21,m23,m31,m32 = pose[0][1],pose[0][2],pose[1][0],pose[1][2],pose[2][0],pose[2][1]
    x,y,z,w = torch.sqrt(m11-m22-m33+1)/2,torch.sqrt(-m11+m22-m33+1)/2,torch.sqrt(-m11-m22+m33+1)/2,torch.sqrt(m11+m22+m33+1)/2
    Quat_ = torch.tensor([
        [x,(m12+m21)/(4*x),(m13+m31)/(4*x),(m23-m32)/(4*x)],
        [(m12+m21)/(4*y),y,(m23+m32)/(4*y),(m31-m13)/(4*y)],
        [(m13 + m31) / (4 * z), (m23 + m32) / (4 * z), z,(m12 - m21) / (4 * z)],
        [(m23 - m32) / (4 * w), (m31 - m13) / (4 * w), (m12 - m21) / (4 * w),w]
    ], dtype=torch.float32)
    _,index = torch.tensor([x,y,z,w]).max(dim=0)
    Quat = Quat_[index.item()]
    return Quat
def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions
def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def iden(x):
    return x

class NeRF_pi(nn.Module):
    def __init__(self, input_dim=3, W=64, pos_multires=10, dir_multires=4):
        super(NeRF_pi,self).__init__()
        self.input_dim = input_dim * (pos_multires * 2 + 1)
        self.input_dir_dim = input_dim * (dir_multires * 2 + 1)
        self.pos_freq_bands = (2. ** torch.linspace(0., pos_multires - 1, steps=pos_multires)) * torch.pi
        self.dir_freq_bands = (2. ** torch.linspace(0., dir_multires - 1, steps=dir_multires)) * torch.pi
        self.W = W
        self.part1 = nn.Sequential(
            nn.Linear(self.input_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU()
        )
        self.part2 = nn.Sequential(
            nn.Linear(self.input_dim+W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )
        self.part3 = nn.Sequential(
            nn.Linear(W + self.input_dir_dim, W),
            nn.ReLU(),
        )

        self.alpha_linear = nn.Sequential(
            nn.Linear(W, 1),
            nn.ReLU(),
        )
        self.rgb_linear = nn.Sequential(
            nn.Linear(W, 3),
            nn.ReLU()
        )

    def get_embed(self,x, freq_bands):
        with torch.no_grad():
            x_ = torch.cat([x*freq for freq in freq_bands], -1)
            x_ = torch.cat([fn(x_) for fn in [torch.sin, torch.cos]], -1).float()
            return torch.cat([x, x_], -1).float()


    def forward(self,pts, viewdirs):
        N_ray, N_sample, _ = pts.shape
        viewdirs = viewdirs[:, None].expand(pts.shape).reshape(-1, 3)
        pts = pts.view(-1, 3)
        gamma = self.get_embed(pts, self.pos_freq_bands)
        dirs = self.get_embed(viewdirs, self.dir_freq_bands)
        out = self.part1(gamma)
        out = torch.cat([out, gamma], -1)
        out1 = self.part2(out)
        alpha = self.alpha_linear(out1).view(N_ray, N_sample, -1)
        out2 = self.part3(torch.cat([out1, dirs], -1)).view(N_ray, N_sample, -1)
        rgb = self.rgb_linear(out2).view(N_ray, N_sample, -1)
        return torch.cat([alpha, rgb], dim=2), out2

class Prd_Net(nn.Module): 
    def __init__(self):
        super(Prd_Net, self).__init__()
        self.atten1 = nn.MultiheadAttention(embed_dim=64*2, num_heads=2, dropout=0.0, batch_first=True)
        self.feed1=nn.Sequential(
            nn.Linear(64*2,64),
            nn.LeakyReLU(0.1,True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_norm1 = nn.LayerNorm([1200,64])
        self.atten2 = nn.MultiheadAttention(embed_dim=64*2, num_heads=2, dropout=0.0, batch_first=True)
        self.feed2 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64,  64),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_norm2 = nn.LayerNorm([1200, 64])
        self.middle = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 , 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, True),
            nn.Linear(32, 1)
        )


    def forward(self, prd_map):

        x = F.avg_pool2d(prd_map, (3, 3), 3, 0).view(prd_map.shape[0], prd_map.shape[1], -1).transpose(1, 2)  # [batch, 1200, 64]

        x_ = torch.cat([x,x],-1)
      
        aten1 = self.atten1(x_,x_, x_, need_weights=False)[0]
        out1 = self.feed1((aten1+x_).reshape(-1,aten1.shape[-1])).reshape(aten1.shape[0],aten1.shape[1],-1)
        out1 = self.layer_norm1(out1)

        out1_ = torch.cat([out1, out1],-1)
        aten2 = self.atten2(out1_, out1_, out1_, need_weights=False)[0]  # [batch, 1200, 64]
        out2 = self.feed2((aten2+out1_).reshape(-1, aten2.shape[-1])).reshape(aten2.shape[0], aten2.shape[1], -1)
        out2 = self.layer_norm2(out2)

        midd = self.middle(out2.mean(dim=1, keepdim=False))

        pred = self.fc(midd)

        return midd, pred

class bypath(nn.Module):
    def __init__(self):
        super(bypath, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.linear = nn.Sequential(
            nn.Linear(1000, 256),
            nn.LeakyReLU(0.1,True),
        )
    def forward(self,x):
        out = self.resnet(x)
        return self.linear(out)

class E2E_model(nn.Module):
    def __init__(self, action_space):
        super(E2E_model,self).__init__()
        self.model_name = 'E2E_model'
        self.bypath = bypath()

        self.linear_process = nn.Sequential(
            nn.Linear(67, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, True),
        )

        self.pred_net = Prd_Net()
        self.policy_net = nn.Sequential(
            nn.Linear(256+64,256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, True),
            nn.Linear(32, action_space)
        )

    def forward(self, x, type = 'gathering', out_pred=None):  # x:[b, 1, 180, 240]
        if type == 'gathering':
            a,b,c,d = out_pred.shape
            pred_in = self.linear_process(out_pred.reshape(-1,out_pred.shape[-1])).view(a,b,c,-1)

            fc,pred = self.pred_net(pred_in.permute([0,3,1,2]))

            out_bypath = self.bypath(x)
            pi = self.policy_net(torch.cat([out_bypath, fc], dim=1))
            return F.softmax(pi, dim=1)
        else:
            a, b, c, d = out_pred.shape
            pred_in = self.linear_process(out_pred.reshape(-1, out_pred.shape[-1])).view(a, b, c, -1)

            fc, pred = self.pred_net(pred_in.permute([0, 3, 1, 2]))
            out_bypath = self.bypath(x)
            pi = self.policy_net(torch.cat([out_bypath, fc], dim=1))
            return F.softmax(pi, dim=1), pred


class NeRF_proc():
    def __init__(self, nerf_tmp, device, nerf_list, Camera_Intrinc='cameras.txt',N_sample=192):
        super(NeRF_proc,  self).__init__()
        self.nerf = nerf_tmp
        self.feature_t = None
        self.N_sample = N_sample
        self.half_dist = 1.0
        self.jitter = True
        self.device = device
        self.nerf = self.nerf.to(self.device)
        self.nerf_list = nerf_list
        with open(Camera_Intrinc, 'r') as f:
            K = f.readline()
            K = K.split(' ')
            self.H, self.W = int(K[-2]), int(K[-1])
            self.K = np.array([
                [float(K[0]), 0, float(K[2])],
                [0, float(K[1]), float(K[3])],
                [0, 0, 1]
            ], dtype=np.float32)
    def change_device(self,device):
        self.device = device
        self.nerf = self.nerf.to(self.device)

    def change_target(self,rgb_t):
        with torch.no_grad():
            self.feature_t = rgb_t[0:-1:2, 0:-1:2, :]
            

    def memory_process(self, drgb, pose, lock, queue, step, nerf_batch=43200, other_device=None):

        depth, rgb = drgb[:, 0:1], drgb[0, 1:, 0:-1:2, 0:-1:2].transpose(0, 1).transpose(1, 2) # depth  [1,1， H,W]， rgb [1,H,W，3]
        
        depths, z_vals = generate_z_vals_and_depths(depth, self.N_sample, self.half_dist, self.jitter) #[H*W,1] [H*W,N_sample, 1]
        pts, viewdirs = generate_rays_half(pose, self.H, self.W, self.K, z_vals) # pts: [N_rays, N_sample, 3],

        if not len(self.nerf_list) == 0:
            lock.acquire()
            _state = self.nerf_list[-1]
            self.nerf.load_state_dict(_state)
            self.nerf = self.nerf.to(self.device)
            self.nerf_list[:] = []
            lock.release()

        raw, out2 = minibatch(nerf_batch, self.nerf, pts, viewdirs)
 
        pred = render_pred(raw, out2, z_vals, self.H // 2, self.W //2, is_flat=False)  

        if step % 3 == 0:
            if other_device==None:
                queue.put([pts, viewdirs, rgb.reshape(-1,3), depths, z_vals])
            else:
                queue.put([pts.to(other_device), viewdirs.to(other_device), rgb.reshape(-1,3).to(other_device), depths.to(other_device), z_vals.to(other_device)])
        return  torch.cat([pred, self.feature_t], dim=-1).unsqueeze(0)


def nerf_reset(nerf, lock, nerf_list):
    for layer in nerf.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
    lock.acquire()
    nerf = nerf.cpu()
    if not len(nerf_list) == 0:
        nerf_list[:] = []
    nerf_list.append(deepcopy(nerf.state_dict()))
    lock.release()

def nerf_train(nerf, N_sample,device, lock, queue, nerf_list, reset_list, child_conn):
    nerf = nerf.to(device)
    nerf.train()
    optimizer = Adam(nerf.parameters(), lr=0.001)
    data_cache = [torch.zeros(0,N_sample,3),torch.zeros(0,3),
                  torch.zeros(0,3),torch.zeros(0,1),torch.zeros(0,N_sample,1),0] # pts, viewdirs, rgb, depths, z_vals, rays_total_nums
    nerf_batch = 10800
    count = 0
    last_undate=0
    H=90
    W=120
    while True:
        if not queue.empty():
            if data_cache[-1] == 0:
                data_cache[0], data_cache[1],data_cache[2], data_cache[3], data_cache[4] = queue.get()
                data_cache[-1] = data_cache[0].shape[0]
            else:
                pts, viewdirs, rgb, depths, z_vals = queue.get()
                data_cache[0] = torch.cat([data_cache[0], pts],0) # N, N_sample, 3
                data_cache[1] = torch.cat([data_cache[1], viewdirs], 0) # N, 3
                data_cache[2] = torch.cat([data_cache[2], rgb], 0)  # N, 3
                data_cache[3] = torch.cat([data_cache[3], depths], 0)  # N, 1
                data_cache[4] = torch.cat([data_cache[4], z_vals], 0) # N, N_sample, 1
                data_cache[-1] += pts.shape[0]

            index = sample(list(range(H*W)), nerf_batch)
            raw, out2 = minibatch(nerf_batch, nerf, data_cache[0][-H*W:][index], data_cache[1][-H*W:][index])
            rgb_map, depth_map = render(raw, data_cache[4][-H*W:][index], H, W, is_flat=False)
            loss_t, psnr = loss(rgb_map, depth_map, data_cache[2][-H*W:][index], data_cache[3][-H*W:][index], True, True)
            # print(loss_t.cpu().item(),psnr)
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            count += 1

            if count-last_undate > 4:
                lock.acquire()
                nerf = nerf.cpu()
                if not len(nerf_list) == 0:
                    nerf_list[:] = []
                nerf_list.append(deepcopy(nerf.state_dict()))
                lock.release()
                nerf = nerf.to(device)
                last_undate = count

        if not data_cache[-1] == 0:
            index = sample(list(range(data_cache[-1])), 1024)
            raw, out2 = minibatch(1024, nerf, data_cache[0][index], data_cache[1][index])
            rgb_map, depth_map = render(raw, data_cache[4][index], is_flat=True)
            loss_t, psnr = loss(rgb_map, depth_map, data_cache[2][index], data_cache[3][index], False, False)

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            count += 1
        if data_cache[-1] > 10800*190:
            data_cache[0] = data_cache[0][10800 * 30:]
            data_cache[1] = data_cache[1][10800 * 30:]
            data_cache[2] = data_cache[2][10800 * 30:]
            data_cache[3] = data_cache[3][10800 * 30:]
            data_cache[4] = data_cache[4][10800 * 30:]
            torch.cuda.empty_cache()
            data_cache[-1] = data_cache[0].shape[0]

        if reset_list[-1]:
            st = 'total train num:%d'%count
            count =0
            last_undate=0
            del data_cache
            data_cache = [torch.zeros(0, N_sample, 3), torch.zeros(0, 3),
                          torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, N_sample, 1),
                          0]  # pts, viewdirs, rgb, depths, z_vals, rays_total_nums
            while not queue.empty():  ### 清空queue
                queue.get()
            reset_list[-1] = False
            # torch.cuda.empty_cache()
            nerf_reset(nerf, lock, nerf_list)
            nerf = nerf.to(device)
            child_conn.send('reset '+st)

def model_train(model, device, lock, source, summary_path, model_path, init_lr,batch_Size,_flag):
    model = model.to(device)
    model.train()
    lr=init_lr = init_lr
    optimizer = Adam(model.parameters(), lr=init_lr)
    writer = SummaryWriter(summary_path)
    flag = False
    num = 0
    batchSize = batch_Size
    total_p, total_t = 0, 0
    _count = 0
    episode = 0
    stats = {'policy_loss':[], 'pred_loss':[], 'learning_rate':[]}
    print(len(source) > 0)
    while True:
        if len(source) > 0:
            lock.acquire()
            _action = np.array([tt.action for tt in source], dtype=np.int64)
            indexl = np.where(_action == 1)[0]
            indexr = np.where(_action == 2)[0]
            indexf = np.where(_action == 0)[0]
            index_l_r = np.concatenate([indexl,indexr],0)
            mean = ((len(indexl) + len(indexr)) * 2) // 5
            index_l_r_ = np.random.choice(index_l_r, mean)
        
            index_ = np.concatenate([indexl, indexr,indexf,index_l_r_], 0)
  
            action = []
            state = []
            out_pred = []
            label = []
            
            for _i in index_:
                action.append(source[_i].action)
                state.append(source[_i].state)
                out_pred.append(source[_i].prd_map)
                label.append(source[_i].label)
                
            action = torch.from_numpy(np.array(action, dtype=np.int64)).view(-1,1).to(device)
            state = torch.from_numpy(np.concatenate(state, 0)).float().to(device)
            out_pred = torch.cat(out_pred, dim=0).to(device)
            label = torch.from_numpy(np.stack(label, 0)).to(device)

            flag = True
            num = label.shape[0]
            print('length:', num)
            source[:] = []
            lock.release()
            torch.cuda.empty_cache()
        if flag:
            _count=0
            total_p, total_t = 0, 0
            for index in BatchSampler(SubsetRandomSampler(range(num)), batchSize, False):
                action_prob,pred = model(x=state[index], out_pred=out_pred[index], type='training')
                action_loss = F.cross_entropy(action_prob, action[index].view(-1))
                theta_loss = F.l1_loss(pred, label[index])
                loss = action_loss + theta_loss
                optimizer.zero_grad()  # Delete old gradients
                loss.backward()  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(model.parameters(), 0.6)  # Clip gradients
                optimizer.step()  # Perform training step bas       ed on gradients
                total_p += action_loss
                total_t += theta_loss
                _count += 1

            episode += 1
            lr = adjust_learning_rate(init_lr,1e6,episode,optimizer)
            stats['policy_loss'].append(total_p.cpu().item()/_count)
            stats['pred_loss'].append(total_t.cpu().item()/_count)
            stats['learning_rate'].append(lr)
            writeSummary(writer,stats, episode)
            if episode % 500 == 0:
                torch.save(model.state_dict(), model_path + str(episode) + '.pkl')



