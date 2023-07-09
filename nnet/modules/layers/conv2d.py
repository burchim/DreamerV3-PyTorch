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

class Conv2d(nn.Conv2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):
        
        super(Conv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding if isinstance(padding, int) or isinstance(padding, tuple) else 0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if isinstance(padding, nn.Module):

            self.pre_padding = padding

        elif isinstance(padding, str):

            # Assert
            assert padding in ["valid", "same", "same-left"]

            # Padding
            if padding == "valid":

                self.pre_padding = nn.Identity()

            elif padding == "same":

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        (self.kernel_size[1] - 1) // 2, # left
                        self.kernel_size[1] // 2, # right
                        
                        (self.kernel_size[0] - 1) // 2, # top
                        self.kernel_size[0] // 2 # bottom
                    ), 
                    value=0
                )

            elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        self.kernel_size[1] // 2,
                        (self.kernel_size[1] - 1) // 2, 

                        self.kernel_size[0] // 2,
                        (self.kernel_size[0] - 1) // 2 
                    ), 
                    value=0
                )
        
        elif isinstance(padding, int) or isinstance(padding, tuple):
            
            self.pre_padding = nn.Identity()

        else:

            raise Exception("Unknown padding: ", padding, type(padding))

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

        # Mask
        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x