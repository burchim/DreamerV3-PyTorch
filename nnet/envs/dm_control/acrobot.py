# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# NeuralNets
from nnet.envs import dm_control

class Acrobot(dm_control.DeepMindControlEnv):

    """
    
    Pendulum (dim(S)=4, dim(A)=1, dim(O)=6): 
    The underactuated double pendulum, torque applied to the second joint. 
    The goal is to swing up and balance. Despite being low-dimensional, this is not an easy control problem. 
    The physical model conforms to (Coulom, 2002) rather than the earlier (Spong, 1995). 
    Both swingup and swingup_sparse tasks with smooth and sparse rewards, respectively.

    Tasks:
    acrobot swingup
    acrobot swingup_sparse

    Reference:
    DeepMind Control Suite, Tassa et al.
    https://arxiv.org/abs/1801.00690
    https://www.youtube.com/watch?v=rAai4QzcYbs
    
    """

    def __init__(self, task="swingup", img_size=(84, 84), mode="classic", history_frames=4, episode_saving_path=None, action_repeat=1):
        assert task in ["swingup", "swingup_sparse"]
        super(Acrobot, self).__init__(domain="acrobot", task=task, img_size=img_size, mode=mode, history_frames=history_frames, episode_saving_path=episode_saving_path, action_repeat=action_repeat)

        self.num_actions = 1
        self.state_size = 6
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Acrobot, self).process_infos(infos)

        # lowd
        obs_orientations = torch.tensor(infos.observation["orientations"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_orientations, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done