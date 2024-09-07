import numpy as np
import os
from random import choice
import torch

def read_episode(root, sc, index,depth_max):
    path = os.path.join(root, sc, index)
    observation = np.load(path+'/observation.npy',allow_pickle=False)
    observation[:, 0] /= depth_max
    action = np.load(path + '/action.npy',allow_pickle=False).astype(np.int64).reshape(-1, 1)
    label = np.load(path + '/theta.npy',allow_pickle=False).reshape(-1, 1)
    robot_T = np.load(path + '/robot_pos_ori.npy',allow_pickle=False).reshape((-1, 4, 4))

    return observation, action, label, robot_T

def dataset(root, device,depth_max):
    scene = os.listdir(root)
    while True:
        sc = choice(scene)
        idd = choice(os.listdir(root+'/'+sc))
        observation, action, label, robot_T = read_episode(root, sc, idd,depth_max)
        print(sc, idd, end=' ')
        target = observation[-1][1:].transpose([1,2,0])
        target = torch.from_numpy(target).to(device)
        for i in range(observation.shape[0]-1):
            done = i==observation.shape[0]-2
            yield observation[i:i+1], robot_T[i], action[i], label[i], target, i, done

