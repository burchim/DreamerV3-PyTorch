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

class Humanoid(dm_control.DeepMindControlEnv):

    """
    
    Humanoid (dim(S)=54, dim(A)=21, dim(O)=67): A simplified humanoid with 21 joints, based on the model in (Tassa et al., 2012). 
    Three tasks: stand, walk and run are differentiated by the desired horizontal speed of 0, 1 and 10m/s, respectively. 
    Observations are in an egocentric frame and many movement styles are possible solutions e.g. running backwards or sideways. This facilitates exploration of local optima.

    humanoid     stand
    humanoid     walk
    humanoid     run
    
    """

    def __init__(self, img_size=(84, 84), mode="classic", history_frames=4, episode_saving_path=None, task="run", action_repeat=1):
        assert task in ["run", "walk", "stand"]
        super(Humanoid, self).__init__(domain="humanoid", task=task, img_size=img_size, mode=mode, history_frames=history_frames, episode_saving_path=episode_saving_path, action_repeat=action_repeat)

        self.num_actions = 21
        self.state_size = (67,)
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Humanoid, self).process_infos(infos)

        # Lowd
        obs_joint_angles = torch.tensor(infos.observation["joint_angles"], dtype=torch.float32)
        obs_head_height = torch.tensor([infos.observation["head_height"]], dtype=torch.float32)
        obs_extremities = torch.tensor(infos.observation["extremities"], dtype=torch.float32)
        obs_torso_vertical = torch.tensor(infos.observation["torso_vertical"], dtype=torch.float32)
        obs_com_velocity = torch.tensor(infos.observation["com_velocity"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_joint_angles, obs_head_height, obs_extremities, obs_torso_vertical, obs_com_velocity, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done