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

class Quadruped(dm_control.DeepMindControlEnv):

    """
    
    Quadruped

    quadruped     walk
    quadruped     run
    quadruped     escape
    quadruped     fetch
    
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
        assert task in ["walk", "run", "escape", "fetch"]
        super(Quadruped, self).__init__(
            domain="quadruped", 
            task=task, 
            img_size=img_size, 
            mode=mode, 
            history_frames=history_frames, 
            episode_saving_path=episode_saving_path, 
            action_repeat=action_repeat, 
            camera_id=2, 
            apply_random_background=apply_random_background, 
            background_videos=background_videos
        )

        self.num_actions = 12
        self.state_size = 78
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Quadruped, self).process_infos(infos)

        # Lowd
        obs_egocentric_state = torch.tensor(infos.observation["egocentric_state"], dtype=torch.float32)
        obs_torso_velocity = torch.tensor(infos.observation["torso_velocity"], dtype=torch.float32)
        obs_torso_upright = torch.tensor(infos.observation["torso_upright"].reshape(1), dtype=torch.float32)
        obs_imu = torch.tensor(infos.observation["imu"], dtype=torch.float32)
        obs_force_torque = torch.tensor(infos.observation["force_torque"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_egocentric_state, obs_torso_velocity, obs_torso_upright, obs_imu, obs_force_torque], dim=0)

        return obs_lowd, obs_pixels, reward, done