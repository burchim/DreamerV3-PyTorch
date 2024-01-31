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

class Swimmer(dm_control.DeepMindControlEnv):

    """
    
    Swimmer (dim(S)=2k+4, dim(A)=k-1, dim(O)=4k+1): 
    This procedurally generated k-link planar swimmer is based on (Coulom, 2002) but using MuJoCo's high- Reynolds fluid drag model. 
    A reward of 1 is provided when the nose is inside the target and decreases smoothly with distance like a Lorentzian. 
    The two instantiations provided in the benchmarking set are the 6-link and 15-link swimmers.
    
    """

    def __init__(
            self, 
            task="swimmer6", 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
            ):
        assert task in ["swimmer6", "swimmer15"]
        super(Swimmer, self).__init__(
            domain="swimmer", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat, 
            apply_random_background=apply_random_background, 
            background_videos=background_videos
        )

        if self.task == "swimmer6":
            self.num_actions = 5
            self.state_size = 25
            self.action_size = (self.num_actions,)
        elif self.task == "swimmer15":
            self.num_actions = 14
            self.state_size = 61
            self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Swimmer, self).process_infos(infos)

        # Lowd
        obs_joints = torch.tensor(infos.observation["joints"], dtype=torch.float32)
        obs_to_target = torch.tensor(infos.observation["to_target"], dtype=torch.float32)
        obs_body_velocities = torch.tensor(infos.observation["body_velocities"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_joints, obs_to_target, obs_body_velocities], dim=0)

        return obs_lowd, obs_pixels, reward, done