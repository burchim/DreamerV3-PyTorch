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

class Hopper(dm_control.DeepMindControlEnv):

    """
    
    Hopper (dim(S)=14, dim(A)=4, dim(O)=15): The planar one-legged hopper introduced in (Lil- licrap et al., 2015), initialised in a random configuration. 
    In the stand task it is rewarded for bringing its torso to a minimal height. 
    In the hop task it is rewarded for torso height and forward velocity.
    
    """

    def __init__(
            self, 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            task="hop", 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
            ):

        assert task in ["hop", "stand"]
        super(Hopper, self).__init__(
            domain="hopper", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat, 
            apply_random_background=apply_random_background, 
            background_videos=background_videos
        )

        self.num_actions = 4
        self.state_size = (15,)
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Hopper, self).process_infos(infos)

        # Lowd
        obs_position = torch.tensor(infos.observation["position"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_touch = torch.tensor(infos.observation["touch"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_position, obs_velocity, obs_touch], dim=0)

        return obs_lowd, obs_pixels, reward, done