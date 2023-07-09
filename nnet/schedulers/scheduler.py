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
import torch.nn as nn

class Scheduler(nn.Module):

    def __init__(self):
        super(Scheduler, self).__init__()

        # Model Step
        self.model_step = torch.tensor(0)

    def step(self):
        self.model_step += 1
        return self.get_val()

    def get_val(self):
        return self.get_val_step(self.model_step)

    def get_val_step(self, step):
        return None