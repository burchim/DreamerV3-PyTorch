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
import glob

class DeepMindControlEnv:

    """
    
    Version:
        dm-control 1.0.9

    Step return:
        state: env state (can be low dim or pixel)
        reward: step reward (0 if reset)
        done: False (agent never dies)
        is_first: True if reset else False
        is_last: False ! need to set True for last step

    """

    def __init__(
            self, 
            domain, 
            task, 
            img_size=(84, 84), 
            mode="classic", 
            history_frames=4, 
            episode_saving_path=None, 
            camera_id=0, 
            action_repeat=1,
            apply_random_background=False,
            background_videos=None
        ):

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
        self.apply_random_background = apply_random_background
        if background_videos is not None:
            self.background_videos = glob.glob(background_videos)
        else:
            self.background_videos = background_videos

        # episode_saving_path
        self.episode_saving_path = episode_saving_path
        if self.episode_saving_path is not None:
            if not os.path.isdir(self.episode_saving_path):
                os.makedirs(self.episode_saving_path, exist_ok=True)

        # Transform
        self.grayscale = torchvision.transforms.Grayscale() if self.mode == "grayscale" else nn.Identity()

        # FPS
        self.fps = 50

    def reset_background(self):
        # (T, C, H, W) load only first 20 seconds
        self.background_video = torchvision.io.read_video(random.choice(self.background_videos), pts_unit="sec", start_pts=0.0, end_pts=20.0)[0].permute(0, 3, 1, 2)
        # print(self.background_video.shape)
        self.background_video = torchvision.transforms.functional.resize(self.background_video, size=self.img_size)
        self.background_t = 0

    def get_background(self, reset=False):

        # Reset video
        if reset:
            self.reset_background()

        background_video = self.background_video[self.background_t]
        self.background_t += 1

        # Reset if video too short
        if self.background_t >= self.background_video.shape[0]:
            self.reset_background()

        return background_video

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
            self.num_channels = obs_pixels.shape[0]
            self.history = obs_pixels.repeat(self.history_frames, 1, 1)
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
            self.history = torch.cat([self.history[3:], obs_pixels], dim=0)
            state = self.history
        
        # Pixel Control Grayscale
        elif self.mode == "grayscale":
            self.history = torch.cat([self.history[1:], obs_pixels], dim=0)
            state = self.history
        
        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

    def process_infos(self, infos):

        # Extract Pixels and convert grayscale if needed
        if "pixels" in infos.observation:
            obs_pixels = self.grayscale(torch.tensor(infos.observation["pixels"].copy()).permute(2, 0, 1)) # uint8, mem efficient for buffer
        else:
            obs_pixels = None

        # Random Background
        if self.apply_random_background:
            assert self.mode == "rgb"
            mask = torch.logical_and((obs_pixels[2] > obs_pixels[1]), (obs_pixels[2] > obs_pixels[0]))
            obs_pixels = torch.where(mask.repeat(3, 1, 1), self.get_background(reset=infos.first()), obs_pixels)

        # Reward
        reward = infos.reward

        # Done
        done = infos.last()

        return obs_pixels, reward, done