import torch
import numpy as np
from collections import namedtuple

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

