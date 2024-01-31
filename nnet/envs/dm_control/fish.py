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

class Fish(dm_control.DeepMindControlEnv):

    """
    
    Fish (dim(S)=26, dim(A)=5, dim(O)=24): 
    A fish is required to swim to a target. 
    This domain relies on MuJoCo's simplified fluid dynamics. 
    Two tasks: in the upright task, the fish is rewarded only for righting itself with respect to the vertical, 
    while in the swim task it is also rewarded for swimming to the target.
    
    """

    def __init__(
            self, 
            task="swim", 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
            ):
        assert task in ["swim", "upright"]
        super(Fish, self).__init__(
            domain="fish", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat, 
            apply_random_background=apply_random_background, 
            background_videos=background_videos
        )

        self.num_actions = 5
        self.state_size = 24
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Fish, self).process_infos(infos)

        # Lowd
        obs_joint_angles = torch.tensor(infos.observation["joint_angles"], dtype=torch.float32)
        obs_upright = torch.tensor([infos.observation["upright"]], dtype=torch.float32)
        obs_target = torch.tensor(infos.observation["target"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_joint_angles, obs_upright, obs_target, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done