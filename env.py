import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig,HabitatSimDepthSensorConfig,EnvironmentConfig,IteratorOptionsConfig
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


def Quat2Rotation(x,y,z,w):
    l1 = np.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],axis=0)
    l2 = np.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],axis=0)
    l3 = np.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2], axis=0)
    T_w = np.stack([l1,l2,l3],axis=0)
    return T_w


class Robot():
    def __init__(self,config,DATA_PATH,SCENES_DIR,success_distance=0.5,split='val',
                 height=480, width=640,hfov=90, max_depth=10,camera_height=1.25,turn_angle=15, forward_step_size=0.25):
        self.success_distance = success_distance
        with read_write(config):
            config.habitat.task.measurements.success.success_distance = success_distance
            config.habitat.dataset.split = split
            config.habitat.dataset.data_path = DATA_PATH
            config.habitat.dataset.scenes_dir = SCENES_DIR

            config.habitat.environment = EnvironmentConfig(max_episode_steps=999999999, max_episode_seconds=9999999999,
                                                           iterator_options=IteratorOptionsConfig(cycle=False,
                                                                                                  shuffle=False,
                                                                                                  group_by_scene=True,
                                                                                                  max_scene_repeat_episodes=1,
                                                                                                  max_scene_repeat_steps=1
                                                                                                  ))
            agent_config = get_agent_config(sim_config=config.habitat.simulator)
            agent_config.sim_sensors.update(
                {"rgb_sensor": HabitatSimRGBSensorConfig(height=height, width=width,position=[0,camera_height,0],hfov=hfov)}
            )
            agent_config.sim_sensors.update(
                {"depth_sensor": HabitatSimDepthSensorConfig(height=height, width=width,
                                                             max_depth=max_depth,position=[0,camera_height,0],hfov=hfov)}
            )
            config.habitat.simulator.turn_angle = turn_angle
            config.habitat.simulator.forward_step_size = forward_step_size

        self.config = config
        self.env = habitat.Env(config=config)
        self.rgb = None
        self.depth = None
        self.target_rgb = None

        self.start_position = None  # start position
        self.start_rotation = None  # the pose of agent at start（quaternion）
        self.target_position = None # target position

        self.robot_pos = None
        self.T = None
        self.theta = None
        self.metrics = None
        self.path_distance=0
        self.last_position=None
    def get_best_action(self, target_position):
        success_distance = self.config.habitat.task.measurements.success.success_distance
        follower = ShortestPathFollower(self.env.sim, success_distance)
        action_pro = np.array(follower.get_next_action(target_position))
        best_action = np.argmax(action_pro)
        return best_action

    def reset(self):
        observations = self.env.reset()
        self.start_position = np.array(self.env.current_episode.start_position)  # start position
        self.start_rotation = self.env.current_episode.start_rotation  # the pose of agent at start（quaternion）

        self.path_distance = 0
        self.last_position = self.start_position

        target_position = self.env.current_episode.goals[0].position  # target position
        self.target_position = np.array([target_position[0],target_position[1],target_position[2],1])
        try:
            self.target_rotation = self.env.current_episode.info['target_rotation']  # x,y,z,w
        except:
            self.target_rotation = np.array([0,0,0,1])
        self.target_rgb = self.env.sim.get_observations_at(self.target_position.tolist()[:3], self.target_rotation)['rgb']
 
        self.rgb = observations['rgb']
        self.depth = observations['depth']


        agent_state = self.env.sim.get_agent_state()
        self.robot_pos = agent_state.position  
        wr, xr, yr, zr = agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z  # 四元数, w,x,-y,z,这里y要取负数才是对的四元数
        T = np.concatenate([Quat2Rotation(xr, -yr, zr, wr), self.robot_pos.reshape(3, 1)], axis=1)
        self.T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

        theta = np.matmul(np.linalg.inv(self.T), np.array(self.target_position).reshape(4, 1))
        theta = theta.reshape(4)[[0, 2]]
        theta /= np.linalg.norm(theta, ord=2)
        theta = np.arctan2(theta[0], theta[1]) / np.pi
        if -1 <= theta < -0.5:
            theta += 1.5
        else:
            theta -= 0.5
        self.theta = theta
        self.metrics = self.env.get_metrics()
        return self.rgb,self.depth,self.target_rgb,self.metrics

    def step(self, act):
        observations = self.env.step(act)
        self.depth = observations['depth']
        self.rgb = observations['rgb']
        agent_state = self.env.sim.get_agent_state()
        self.robot_pos = agent_state.position  
        self.path_distance += np.linalg.norm(self.robot_pos-self.last_position,ord=2)
        self.last_position = self.robot_pos

        wr, xr, yr, zr = agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z  # 四元数, w,x,-y,z,这里y要取负数才是对的四元数
        T = np.concatenate([Quat2Rotation(xr, -yr, zr, wr), self.robot_pos.reshape(3, 1)], axis=1)
        self.T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

        theta = np.matmul(np.linalg.inv(self.T), self.target_position.reshape(4, 1))
        theta = theta.reshape(4)[[0, 2]]
        theta /= np.linalg.norm(theta, ord=2)
        theta = np.arctan2(theta[0], theta[1]) / np.pi
        if -1 <= theta < -0.5:
            theta += 1.5
        else:
            theta -= 0.5
        self.theta = theta
        self.metrics = self.env.get_metrics()
        return self.rgb,self.depth, self.metrics

