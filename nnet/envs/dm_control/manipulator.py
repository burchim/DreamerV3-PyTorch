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

class Manipulator(dm_control.DeepMindControlEnv):

    """
    
    Manipulator (dim(S)=22, dim(A)=5, dim(O)=37): 
    A planar manipulator is rewarded for bringing an object to a target location. 
    In order to assist with exploration, in %10 of episodes the object is initialised in the gripper or at the target. 
    Four manipulator tasks: {bring,insert}_{ball,peg} of which only bring_ball is in the benchmarking set.
    
    """

    def __init__(
            self, 
            task="bring_ball", 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
            ):
        assert task in ["bring_ball"]
        super(Manipulator, self).__init__(
            domain="manipulator", 
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
        self.state_size = 44
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(Manipulator, self).process_infos(infos)

        # Lowd
        obs_arm_pos = torch.tensor(infos.observation["arm_pos"].reshape(-1), dtype=torch.float32)
        obs_arm_vel = torch.tensor(infos.observation["arm_vel"], dtype=torch.float32)
        obs_touch = torch.tensor(infos.observation["touch"], dtype=torch.float32)
        obs_hand_pos = torch.tensor(infos.observation["hand_pos"], dtype=torch.float32)
        obs_object_pos = torch.tensor(infos.observation["object_pos"], dtype=torch.float32)
        obs_object_vel = torch.tensor(infos.observation["object_vel"], dtype=torch.float32)
        obs_target_pos = torch.tensor(infos.observation["target_pos"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_arm_pos, obs_arm_vel, obs_touch, obs_hand_pos, obs_object_pos, obs_object_vel, obs_target_pos], dim=0)

        return obs_lowd, obs_pixels, reward, done