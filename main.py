import argparse
import os
import sys
from habitat.config import read_write
import habitat
import json
import numpy as np
from env import Robot
from util import work, Transition
from model import E2E_model, NeRF_proc, model_train, nerf_train, NeRF_pi
from dataset import dataset
import warnings
import torch
from datetime import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe, Queue
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', help='train/test mode')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--dataset', default='../dataset', help='Path to dataset directory ')
parser.add_argument('--checkpoint_path', default = './', help='Path to the saved checkpoint')
parser.add_argument('--N_sample', default = 192, help='The number of ray samples')
parser.add_argument('--episodeLimit', default = 1e8, help='Training epoch limit')
parser.add_argument('--depth_max', default = 10, help='The farthest distance captured by the depth camera')
parser.add_argument('--model_save_name', default = 'model', help='Model name to save')
parser.add_argument('--model_load', default = None, help='The path of the model parameters to import')
parser.add_argument('--action_space', default = 3, help='Action space')
parser.add_argument('--Camera_Intrinc', default = 'cameras.txt', help='The file records the camera intrinc parameters')
parser.add_argument('--nerf_batch', default = 10800, help='Rays number for nerf training')
parser.add_argument('--batch_size', default = 256, help='Batch size for model training')
parser.add_argument('--init_lr', default = 5e-4, help='Initial learning rate')
parser.add_argument('--height', default = 180, help='Image height')
parser.add_argument('--width', default = 240, help='Image width')
parser.add_argument('--hfov', default = 90, help='Image hfov')
parser.add_argument('--camera_height', default = 1.25, help='Camera height in simulator')
parser.add_argument('--turn_angle', default = 15, help='Robot turning angle in simulator')
parser.add_argument('--forward_step_size', default = 0.15, help='Robot stepping forward size in simulator')
parser.add_argument('--success_distance', default = 0.5, help='The distance at which success is judged')
parser.add_argument('--max_step', default = 700, help='Robot max step in test mode')
parser.add_argument('--DATA_PATH', default = None, help='Data path for habitat simulator')
parser.add_argument('--SCENES_DIR', default = None, help='Scenes dir for habitat simulator')
parser.add_argument('--split', default = None, help='Use for simulator environment dataset')
FLAGS = parser.parse_args()

MODE = FLAGS.mode
episodeLimit = FLAGS.episodeLimit
N_sample = FLAGS.N_sample
device = 'cuda:{}'.format(FLAGS.gpu)
checkpoint_path = FLAGS.checkpoint_path
model_save_name = FLAGS.model_save_name
model_load = FLAGS.model_load
dataset_root = FLAGS.dataset
action_space = FLAGS.action_space
nerf_batch = FLAGS.nerf_batch
depth_max = FLAGS.depth_max
init_lr = FLAGS.init_lr
batch_size = FLAGS.batch_size
Camera_Intrinc = FLAGS.Camera_Intrinc
height = FLAGS.height
width = FLAGS.width
hfov = FLAGS.hfov
camera_height = FLAGS.camera_height
turn_angle = FLAGS.turn_angle
forward_step_size = FLAGS.forward_step_size
max_step = FLAGS.max_step
success_distance = FLAGS.success_distance
DATA_PATH = FLAGS.DATA_PATH
SCENES_DIR = FLAGS.SCENES_DIR
split = FLAGS.split

def main(mode):
    model = E2E_model(action_space)
    if not model_load==None: model.load_state_dict(torch.load(model_load))
    nerf = NeRF_pi()   
    nerf_tmp = NeRF_pi()

    mp.set_start_method('spawn')
    manager = mp.Manager()
    source = manager.list()
    print(len(source))
    _flag = manager.list()
    _flag.append(1)
    source_lock = mp.Lock()
    lock = mp.Lock()
    reset_list = manager.list()
    reset_list.append(False)
    parent_conn, child_conn = Pipe()
    nerf_list = manager.list()
    queue = Queue(maxsize=200)

    if mode == 'train':
        date=str(datetime.today())[:10]+'_' + model_save_name
        model_path = os.path.join(checkpoint_path,date,'model')
        summary_path = os.path.join(checkpoint_path,date,'summary')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        
        nerf_proc = NeRF_proc(nerf_tmp, device, nerf_list, Camera_Intrinc,N_sample=N_sample)
        process = []
        p0 = Process(target=model_train, args=(model, device, source_lock, source, summary_path, model_path,init_lr, batch_size, _flag,))
        p0.start()
        process.append(p0)
        p1 = Process(target=nerf_train, args=(nerf, nerf_proc.N_sample, device, lock, queue, nerf_list, reset_list, child_conn,))
        p1.start()
        process.append(p1)
        data = dataset(dataset_root, device,depth_max)
        cache=[]
        episodeCount = 0
        if mode=='train':
            while episodeCount < episodeLimit:
                for observation, robot_T, action, label, target, step, done in data:
                    
                    if step==0: nerf_proc.change_target(target)
                    prd_map = work(nerf_proc, observation, robot_T, lock, queue, step, nerf_batch, device)
                    observation[0, 0] *= depth_max
                    observation[0, 0][observation[0, 0] >= depth_max//2] = depth_max//2
                    observation[0, 0] /= depth_max//2
                    trans = Transition(observation, action, prd_map, label)
                    cache.append(trans)

                    if done:
                        break

                if len(cache) > 15:
                    source_lock.acquire()
                    source.extend(cache)
                    source_lock.release()
                    print(len(cache))
                    cache.clear()

                reset_list[-1] = True
                print(parent_conn.recv(), 'step: %d'%step)  

    elif mode == 'test':
        config = habitat.get_config(config_path="benchmark/nav/pointnav/pointnav_gibson.yaml")
        nerf_proc = NeRF_proc(nerf_tmp, device, nerf_list, Camera_Intrinc,N_sample=N_sample)
        process = []
        p1 = Process(target=nerf_train, args=(nerf, nerf_proc.N_sample, device, lock, queue, nerf_list, reset_list, child_conn,))
        p1.start()
        model = model.to(device)
        model.eval()
        nerf_proc.change_device(device)
        ###### habitat env #######
        robot = Robot(config, DATA_PATH, SCENES_DIR, success_distance=success_distance, split=split,
                              height=height, width=width, max_depth=depth_max, camera_height=camera_height, 
                              hfov=hfov, turn_angle=turn_angle, forward_step_size=0.15)
        ###### test ######
        for episode in range(len(robot.env.episodes)):
            success = 0
            rgb, depth, target_rgb, metrics = robot.reset()
            nerf_proc.change_target(torch.from_numpy(target_rgb.astype(np.float32)/255).to(device))
            observation = np.concatenate([depth, rgb/255], -1).transpose([2, 0, 1]).astype(np.float32)
            observation = np.expand_dims(observation, axis=0)
            robot_T = robot.T.astype(np.float32)
            for step in range(max_step):
                    prd_map = work(nerf_proc, observation, robot_T, lock, queue, step, nerf_batch, device, device)
                    with torch.no_grad():
                        observation[0, 0] *= depth_max
                        observation[0, 0][observation[0, 0] >= depth_max//2] = depth_max//2
                        observation[0, 0] /= depth_max//2
                        action_prob = model(x=torch.from_numpy(observation).to(device), type='gathering',out_pred=prd_map)
                        act = np.argmax(action_prob.cpu()).item() + 1
                    rgb, depth, metrics = robot.step(act)
                    observation = np.concatenate([depth, rgb/255], -1).transpose([2, 0, 1]).astype(np.float32)
                    observation = np.expand_dims(observation, axis=0)
                    robot_T = robot.T.astype(np.float32)
                    if metrics['distance_to_goal'] <= success_distance:
                        success = 1
                        rgb, depth, metrics = robot.step(0)
                        break
            print('step:%d, if_success:%d'%(step,success))

if __name__ == '__main__':
    main(MODE)