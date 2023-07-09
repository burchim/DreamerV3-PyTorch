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
from nnet import inits
from nnet import modules


class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        channels_last=False,
        weight_init="default",
        bias_init="default"
    ):

        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding, 
            groups=groups, 
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.input_permute = modules.PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = modules.PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

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

    def forward(self, x):

        # Apply Weight
        x = self.output_permute(super(ConvTranspose2d, self).forward(self.input_permute(x)))

        return x