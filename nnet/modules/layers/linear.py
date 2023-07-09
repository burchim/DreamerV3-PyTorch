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
import torch.nn.functional as F

# NeuralNets
from nnet import inits
from nnet import modules

class Linear(nn.Linear, modules.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, weight_init="default", bias_init="default", vn_std=None, record_norm=False):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        # Variational Noise
        self.vn_std = vn_std

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                inits.init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                inits.init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                inits.init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                inits.init_dict[bias_init](self.bias)

        # record_norm
        self.record_norm = record_norm

    def forward(self, x):

        # Weight
        weight = self.weight

        # Add Weight Noise
        if self.vn_std is not None and self.training:
            weight = weight + self.vn_std * torch.randn_like(self.weight)
            
        # Apply Weight
        x = F.linear(x, weight, self.bias)

        # record_norm
        if self.record_norm:
            self.add_info("{}.l2_norm".format(self.name), round(x.norm().item(), 4))

        return x