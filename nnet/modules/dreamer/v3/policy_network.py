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
from nnet import modules
from nnet import distributions

class PolicyNetwork(nn.Module):

    def __init__(
            self, 
            num_actions, 
            hidden_size=1024, 
            act_fun=nn.SiLU, 
            num_mlp_layers=5, 
            feat_size=32*32+1024, 
            discrete=False, 
            min_std=0.1, 
            max_std=1.0, 
            uniform_mix=0.01, 
            weight_init="dreamerv3_normal", 
            bias_init="zeros", 
            norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
            dist_weight_init="xavier_uniform", 
            dist_bias_init="zeros"
        ):
        super(PolicyNetwork, self).__init__()

        self.mlp = modules.MultiLayerPerceptron(dim_input=feat_size, dim_layers=[hidden_size for _ in range(num_mlp_layers)], act_fun=act_fun, weight_init=weight_init, bias_init=bias_init, norm=norm, bias=norm is None)
        self.linear_proj = modules.Linear(hidden_size, num_actions if discrete else 2 * num_actions, weight_init=dist_weight_init, bias_init=dist_bias_init)
        self.min_std = min_std
        self.max_std = max_std
        self.discrete = discrete
        self.uniform_mix = uniform_mix

    def forward(self, x):

        # MLP Layers
        x = self.mlp(x)

        if self.discrete:

            # Logits Projection
            logits = self.linear_proj(x)

            # One Hot Distribution
            action_dist = distributions.OneHotDist(logits=logits, uniform_mix=self.uniform_mix)

            return action_dist

        else:

            # Mean / Std Proj
            mean, std =  torch.chunk(self.linear_proj(x), chunks=2, dim=-1)

            # Scale mean
            mean = F.tanh(mean)

            # Born std to [self.min_std:self.max_std]
            std = (self.max_std - self.min_std) * F.sigmoid(std + 2.0) + self.min_std

            # Normal Distribution
            action_dist = torch.distributions.Independent(distributions.Normal(mean, std), 1)

            return action_dist
    