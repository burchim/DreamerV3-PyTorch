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
from nnet.structs import AttrDict

# Gym
import gym

class MemoryMaze:

    """
    
    keys,           Shape,              Type,       Description

    image,          (64, 64, 3),        uint8,      First-person view observation
    action,         (6),                binary,     Last action, one-hot encoded
    reward,         (),                 float,      Last reward
    
    maze_layout,    (9, 9) | (15, 15),  binary,     Maze layout (wall / no wall)
    agent_pos,      (2),                float,      Agent position in global coordinates
    agent_dir,      (2),                float,      Agent orientation as a unit vector
    targets_pos,    (3, 2) | (6, 2),    float,      Object locations in global coordinates
    targets_vec,    (3, 2) | (6, 2),    float,      Object locations in agent-centric coordinates
    target_pos,     (2),                float,      Current target object location, global
    target_vec,     (2),                float,      Current target object location, agent-centric
    target_color,   (3),                float,      Current target object color RGB
    
    """

    def obs_space(self):

        return (["image", (3, 64, 64), torch.float32],)

    def __init__(
            self,
            maze_size="15x15"
    ):
        
        # desactivate logging: "WARNING:absl:Cannot set velocity on Entity with no free joint."
        from absl import logging
        logging.set_verbosity(logging.ERROR)
        
        self.env_ids = {
            "9x9": "memory_maze:MemoryMaze-9x9-v0",
            "11x11": "memory_maze:MemoryMaze-11x11-v0",
            "13x13": "memory_maze:MemoryMaze-13x13-v0",
            "15x15": "memory_maze:MemoryMaze-15x15-v0",
        }

        self.num_actions = 6 # (noop, forward, left, right, forward_left, forward_right)

        self.action_repeat = 1

        self.env = gym.make(self.env_ids[maze_size])

        self.fps = 4

    def sample(self):

        return torch.tensor(self.env.action_space.sample())

    def process_obs(self, obs):

        return torch.tensor(obs).permute(2, 0, 1) # (3, 64, 64)

    def reset(self):

        # Reset Env
        obs = self.env.reset()

        # State
        state = self.process_obs(obs)

        # Reward
        reward = torch.tensor(0.0, dtype=torch.float32)

        # Done
        done = torch.tensor(False, dtype=torch.float32)

        # Is First
        is_first = torch.tensor(True, dtype=torch.float32)

        # Is_last
        is_last = torch.tensor(False, dtype=torch.float32)

        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)
    
    def step(self, action):

        # Step Env
        obs, reward, done, infos = self.env.step(action.item())

        # State
        state = self.process_obs(obs)

        # Reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Done
        done = torch.tensor(done, dtype=torch.float32)

        # Is First
        is_first = torch.tensor(False, dtype=torch.float32)

        # Is_last
        is_last = done

        return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)