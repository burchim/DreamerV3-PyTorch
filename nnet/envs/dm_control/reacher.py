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

class Reacher(dm_control.DeepMindControlEnv):

    """
    
    Reacher (dim(S)=4, dim(A)=2, dim(O)=7): 
    The simple two-link planar reacher with a ran- domised target location. 
    The reward is one when the end effector pen- etrates the target sphere. 
    In the easy task the target sphere is bigger than on the hard task (shown on the left).

    Reacher     easy
    Reacher     hard
    
    """

    def __init__(self, img_size=(84, 84), mode="classic", history_frames=4, episode_saving_path=None, task="easy", action_repeat=1):
        assert task in ["easy", "hard"]
        super(Reacher, self).__init__(domain="reacher", task=task, img_size=img_size, mode=mode, history_frames=history_frames, episode_saving_path=episode_saving_path, action_repeat=action_repeat)

        self.num_actions = 2
        self.state_size = 6
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Reacher, self).process_infos(infos)

        # Lowd
        obs_position = torch.tensor(infos.observation["position"], dtype=torch.float32)
        obs_to_target = torch.tensor(infos.observation["to_target"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_position, obs_to_target, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done