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
from torch import nn
import torchvision

# NeuralNets
from nnet.structs import AttrDict

# DM Control
import os
from dm_control import suite
from dm_control.suite.wrappers import pixels

# Other
import datetime
import random
import sys

class DeepMindControlEnv:

    """
    
    Version:
        dm-control 1.0.9

    """

    def __init__(self, domain, task, img_size=(84, 84), mode="classic", history_frames=4, episode_saving_path=None, camera_id=0, action_repeat=1, norm_mean=(127.5, 127.5, 127.5), norm_std=(255, 255, 255)):

        # Mode
        assert mode in ["classic", "rgb", "grayscale"]
        self.mode = mode

        # Env
        self.camera_id = camera_id
        self.img_size = img_size
        self.domain = domain
        self.task = task
        self.history_frames = history_frames
        self.clip_low = -1
        self.clip_high = 1
        self.action_repeat = action_repeat
        if self.mode in ["rgb", "grayscale"] or episode_saving_path is not None:
            self.env = pixels.Wrapper(suite.load(self.domain, self.task), pixels_only=False, render_kwargs={"height": self.img_size[0], "width": self.img_size[1], "camera_id": self.camera_id})
        else:
            self.env = suite.load(self.domain, self.task)

        # episode_saving_path
        self.episode_saving_path = episode_saving_path
        if self.episode_saving_path is not None:
            if not os.path.isdir(self.episode_saving_path):
                os.makedirs(self.episode_saving_path, exist_ok=True)

        # Transform
        self.grayscale = torchvision.transforms.Grayscale() if self.mode == "grayscale" else nn.Identity()
        self.normalize = torchvision.transforms.Normalize(mean=norm_mean, std=norm_std) # default [-0.5: 0.5]

        # FPS
        self.fps = 50

    def sample(self):

        action = torch.zeros(self.num_actions)
        action.uniform_(self.clip_low, self.clip_high)
        
        return action

    def set_seed(self, seed):
        self.env.seed(seed)

    def random_seed(self):
        self.set_seed(random.randint(0, sys.maxsize))

    def reset(self):

        # Reset Env
        obs_lowd, obs_pixels, _, _ = self.process_infos(self.env.reset())
        
        # Reward
        reward = torch.tensor(0.0, dtype=torch.float32)

        # Done
        done = torch.tensor(False, dtype=torch.float32)

        # Is First
        is_first = torch.tensor(True, dtype=torch.float32)

        # Is_last
        is_last = torch.tensor(False, dtype=torch.float32)

        # Episode Score
        self.episode_score = 0.0

        # Episode videos
        if self.episode_saving_path is not None:
            self.episode_video = []
            self.episode_video_pre = []

        # Classic Control
        if self.mode == "classic":
            state = obs_lowd
        
        # Pixel Control
        elif self.mode in ["rgb", "grayscale"]:
            obs_pixels_normalized = self.normalize(obs_pixels)
            self.num_channels = obs_pixels_normalized.shape[0]
            self.history = obs_pixels_normalized.repeat(self.history_frames, 1, 1)
            state = self.history
        
        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

        
    def get_obs(self):
        return torch.tensor(self.env._env.physics.render(**{"camera_id": self.camera_id}).copy())

    def step(self, action):

        # Assert
        assert action.shape == self.action_size

        # Clip Actions
        action = action.clip(self.clip_low, self.clip_high)

        # Forward Env with action repeat
        reward = 0.0
        done = False
        for i in range(self.action_repeat):

            # Env step
            obs_lowd, obs_pixels, step_reward, step_done = self.process_infos(self.env.step(action.tolist()))

            # Add to video
            if self.episode_saving_path is not None:
                self.episode_video.append(self.get_obs())
                self.episode_video_pre.append(obs_pixels.permute(1, 2, 0))

            # Update reward / done
            reward += step_reward
            done = done or step_done

        # Update Episode Score
        self.episode_score += reward

        # Save Episode
        if done and self.episode_saving_path is not None:

            # Stack videos
            self.episode_video = torch.stack(self.episode_video, dim=0)
            self.episode_video_pre = torch.stack(self.episode_video_pre, dim=0)

            # Datetime
            date_time_score = str(datetime.datetime.now()).replace(" ", "_") + "_" + str(self.episode_score)

            # Save Videos
            torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}.mp4".format(date_time_score)), video_array=self.episode_video, fps=self.fps, video_codec="libx264")
            torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}_pre.mp4".format(date_time_score)), video_array=self.episode_video_pre.unsqueeze(dim=-1).repeat(1, 1, 1, 3) if self.mode == "grayscale" else self.episode_video_pre, fps=self.fps, video_codec="libx264")

        # Reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Done
        done = torch.tensor(False, dtype=torch.float32)

        # Is_last
        is_last = done

        # Is First
        is_first = torch.tensor(False, dtype=torch.float32)

        # Classic Control
        if self.mode == "classic":
            state = obs_lowd
        
        # Pixel COntrol RGB
        elif self.mode == "rgb":
            self.history = torch.cat([self.history[3:], self.normalize(obs_pixels)], dim=0)
            state = self.history
        
        # Pixel Control Grayscale
        elif self.mode == "grayscale":
            self.history = torch.cat([self.history[1:], self.normalize(obs_pixels)], dim=0)
            state = self.history
        
        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

    def process_infos(self, infos):

        # Extract Pixels and convert grayscale if needed
        if "pixels" in infos.observation:
            obs_pixels = self.grayscale(torch.tensor(infos.observation["pixels"].copy(), dtype=torch.float32).permute(2, 0, 1))
        else:
            obs_pixels = None

        # Reward
        reward = infos.reward

        # Done
        done = infos.last()

        return obs_pixels, reward, done