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

# NeuralNets
from nnet import modules
from nnet.utils import get_module_and_params

class ConvNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides=1, norm=None, act_fun="ReLU", drop_rate=0.0, padding="same", dim=2, channels_last=False, residual=False, weight_init="default", bias_init="default", bias=True):
        super(ConvNeuralNetwork, self).__init__()

        # Convs
        conv = {
            2: modules.Conv2d,
        }

        # Get act_fun and norm
        act_fun, act_fun_params = get_module_and_params(act_fun, modules.act_dict)
        norm, norm_params = get_module_and_params(norm, modules.norm_dict)

        # Params
        self.strides = strides
        self.residual = residual

        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]

        # CNN Layers
        self.layers = nn.ModuleList([nn.Sequential(

            # Conv
            conv[dim](
                dim_input if layer_id == 0 else dim_layers[layer_id - 1], 
                dim_layers[layer_id], 
                kernel_size[layer_id] if isinstance(kernel_size, list) else kernel_size, 
                stride=strides[layer_id] if isinstance(strides, list) else strides, 
                padding=padding[layer_id] if isinstance(padding, list) else padding, 
                channels_last=channels_last, 
                weight_init=weight_init[layer_id] if isinstance(weight_init, list) else weight_init, 
                bias_init=bias_init[layer_id] if isinstance(bias_init, list) else bias_init, 
                bias=bias[layer_id] if isinstance(bias, list) else bias
            ), 
            
            # Norm
            norm[layer_id](dim_layers[layer_id], **norm_params[layer_id], channels_last=channels_last) if isinstance(norm, list) else norm(dim_layers[layer_id], **norm_params, channels_last=channels_last),
            
            # Act fun
            act_fun[layer_id](**act_fun_params[layer_id]) if isinstance(act_fun, list) else act_fun(**act_fun_params),
            
            # Dropout
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x):

        # Layers
        for layer in self.layers:

            # Forward
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x