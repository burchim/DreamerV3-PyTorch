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

class TimeLimit:

    def __init__(self, env, time_limit):

        self.env = env
        self.time_limit = time_limit

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):

        # Env Step
        obs = self.env.step(action)

        # Update time step
        self.time_step += self.env.action_repeat

        # Time Limit
        if self.time_step >= self.time_limit:
            obs.is_last = torch.tensor(True, dtype=torch.float32)

        return obs
    
    def reset(self):
        self.time_step = 0
        return self.env.reset()