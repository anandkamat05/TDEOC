import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
from gym.envs.registration import register
import numpy as np


class TwoDEnv(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip, xbounds, ybounds):
        super(TwoDEnv, self).__init__(model_path=model_path, frame_skip=frame_skip)
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)
        
    def get_viewer(self):
        return self._get_viewer()

import numpy as np
from gym import utils
import os




def get_asset_xml(xml_name):
    return os.path.join(os.path.join(os.path.dirname(__file__), 'assets'), xml_name)
    
class TMaze(TwoDEnv, utils.EzPickle):
    NAME='TMaze'
    def __init__(self, verbose=False,change_goal=None):
        self.verbose = verbose
        self.steps = 0
        self.change_goal = change_goal
        self.target_count = [0,0]
        utils.EzPickle.__init__(self)
        TwoDEnv.__init__(self, get_asset_xml('twod_tmaze.xml'), 2, xbounds=[-0.3,0.3], ybounds=[-0.3,0.3])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        
        # if not self.change_goal:
        target = self.model.body_pos.copy()[-2][:2]
        target_2 = self.model.body_pos.copy()[-1][:2]
        # else:
        #     target = self.change_goal

        dist_thresh = 0.1

        if self.near(pos, target):
            if not self.change_goal:
                self.target_count[0] += 1
                reward = 1.
            elif np.array_equal(self.change_goal, target):
                self.target_count[0] += 1
                reward = 1.
            else:
                reward = 0
        elif self.near(pos, target_2):
            if not self.change_goal:
                self.target_count[1] += 1
                reward = 1.
            elif np.array_equal(self.change_goal, target_2):
                self.target_count[1] += 1
                reward = 1.
            else:
                reward = 0
        else:
            reward = 0.

        self.steps += 1
        if self.verbose:
            print(pos, reward)
        done = self.steps >= 500 or int(reward)
        return ob, reward, done, np.concatenate([self.data.qvel]).ravel()

    def change(self):
        if self.target_count[0] >= self.target_count[1]:
            self.change_goal = [-.3,.3]
        else:
            self.change_goal = [.3,.3]

    def near(self, pos, target, dist_thresh=0.1):
        if pos[0]>target[0]-dist_thresh and pos[0]<target[0]+dist_thresh \
                and pos[1]<target[1]+dist_thresh and pos[1]>target[1]-dist_thresh:
            return True
        else:
            return False

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        init_pos = self.model.body_pos.copy()[1][:2]
        return np.concatenate([self.data.qpos]).ravel() + init_pos

    def viewer_setup(self):
        v = self.viewer
