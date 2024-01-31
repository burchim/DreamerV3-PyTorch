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

class BallInCup(dm_control.DeepMindControlEnv):

    """
    
    Ball in cup (dim(S)=8, dim(A)=2, dim(O)=8): 
    A planar ball-in-cup task. An actuated planar receptacle can translate 
    in the vertical plane in order to swing and catch a ball attached to its bottom. 
    The catch task has a sparse reward: 1 when the ball is in the cup, 0 otherwise.

    Tasks:
    ball_in_cup catch

    Reference:
    DeepMind Control Suite, Tassa et al.
    https://arxiv.org/abs/1801.00690
    https://www.youtube.com/watch?v=rAai4QzcYbs
    
    """

    def __init__(
            self, 
            task="catch", 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
        ):
        assert task in ["catch"]
        super(BallInCup, self).__init__(
            domain="ball_in_cup", 
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
        self.state_size = 8
        self.action_size = (self.num_actions,)

    def process_infos(self, infos):
        obs_pixels, reward, done = super(BallInCup, self).process_infos(infos)

        # lowd
        obs_position = torch.tensor(infos.observation["position"], dtype=torch.float32)
        obs_velocity = torch.tensor(infos.observation["velocity"], dtype=torch.float32)
        obs_lowd = torch.cat([obs_position, obs_velocity], dim=0)

        return obs_lowd, obs_pixels, reward, done