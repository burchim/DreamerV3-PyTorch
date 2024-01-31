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

class Walker(dm_control.DeepMindControlEnv):

    """
    
    Walker (dim(S)=18, dim(A)=6, dim(O)=24): An improved planar walker based on the one introduced in (Lillicrap et al., 2015). 
    In the stand task reward is a combination of terms encouraging an upright torso and some minimal torso height. 
    The walk and run tasks include a component encouraging forward velocity.

    tasks: run, stand, walk
    
    """

    def __init__(
            self, 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            task="run", 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
        ):
        assert task in ["run", "stand", "walk"]
        super(Walker, self).__init__(
            domain="walker", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat,
            apply_random_background=apply_random_background,
            background_videos=background_videos
        )

        # Env Params
        self.num_actions = 6
        self.state_size = 24
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Walker, self).process_infos(infos)

        # Lowd
        obs_orientations = torch.tensor(infos.observation["orientations"], dtype=torch.float32)
        obs_height = torch.tensor([infos.observation["height"]], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_orientations, obs_height, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done