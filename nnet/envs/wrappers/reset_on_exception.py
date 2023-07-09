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

class ResetOnException:

    def __init__(self, env, verbose=1):

        self.env = env
        self.verbose = verbose

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):

        # Env Step
        try:
            obs = self.env.step(action)
            obs.error = torch.tensor(False, dtype=torch.float32)

        # Reset On Exception
        except Exception as e:
            if self.verbose:
                print("Restarting env after crash with: {}".format(e), flush=True)
            obs = self.env.reset()
            obs.error = torch.tensor(True, dtype=torch.float32)

        return obs
    
    def reset(self):
        return self.env.reset()