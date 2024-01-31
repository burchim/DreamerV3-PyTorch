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

class Finger(dm_control.DeepMindControlEnv):

    """
    
    Pendulum (dim(S)=6, dim(A)=2, dim(O)=12): 
    A 3-DoF toy manipulation problem based on (Tassa and Todorov, 2010). 
    A planar 'finger' is required to rotate a body on an unactuated hinge. 
    In the turn_easy and turn_hard tasks, the tip of the free body must overlap with a target (the target is smaller for the turn_hard task). 
    In the spin task, the body must be continually rotated.

    Reference:
    DeepMind Control Suite, Tassa et al.
    https://arxiv.org/abs/1801.00690
    https://www.youtube.com/watch?v=rAai4QzcYbs
    
    """

    def __init__(
            self, 
            task="spin", 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
            ):
        assert task in ["spin", "turn_easy", "turn_hard"]
        super(Finger, self).__init__(
            domain="finger", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat,
            apply_random_background=apply_random_background,
            background_videos=background_videos
        )

        self.num_actions = 2
        if self.task == "spin":
            self.state_size = 9
        else:
            self.state_size = 12
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Finger, self).process_infos(infos)

        # lowd
        obs_position = torch.tensor(infos.observation["position"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_touch = torch.tensor(infos.observation["touch"], dtype=torch.float32)
        if self.task == "spin":
            obs_lowd = torch.cat([obs_position, obs_velocity, obs_touch], dim=0)
        else:
            obs_target_position = torch.tensor(infos.observation["target_position"], dtype=torch.float32)
            obs_dist_to_target = torch.tensor([infos.observation["dist_to_target"]], dtype=torch.float32)
            obs_lowd = torch.cat([obs_position, obs_velocity, obs_touch, obs_target_position, obs_dist_to_target], dim=0)

        return obs_lowd, obs_pixels, reward, done