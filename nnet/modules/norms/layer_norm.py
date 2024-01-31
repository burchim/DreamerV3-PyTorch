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
from nnet.modules import layers

class LayerNorm(nn.LayerNorm):

    def __init__(
            self, 
            normalized_shape, 
            eps=1e-05, 
            elementwise_affine=True, 
            device=None, 
            dtype=None, 
            channels_last=True, 
            flatten_start_end_dims=None,
            convert_float32=False
        ):
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape, 
            eps=eps, 
            elementwise_affine=elementwise_affine, 
            device=device, 
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.permute_input = nn.Identity()
            self.permute_output = nn.Identity()
        else:
            self.permute_input = layers.PermuteChannels(to_last=True)
            self.permute_output = layers.PermuteChannels(to_last=False)

        # Flatten / Reshape dims before / after norm
        self.flatten_start_end_dims = flatten_start_end_dims

        # convert_float32
        self.convert_float32 = convert_float32

    def forward(self, input):

        # Flatten input
        if self.flatten_start_end_dims is not None:
            input_shape = input.shape
            input = input.flatten(start_dim=self.flatten_start_end_dims[0], end_dim=self.flatten_start_end_dims[1])

        # LayerNorm
        if self.convert_float32 and input.dtype != torch.float32:
            with torch.cuda.amp.autocast(enabled=False):
                output = self.permute_output(super(LayerNorm, self).forward(self.permute_input(input.type(torch.float32)))).type(input.dtype)
        else:
            output = self.permute_output(super(LayerNorm, self).forward(self.permute_input(input)))

        # Reshape output
        if self.flatten_start_end_dims is not None:
            output = output.reshape(input_shape)

        return output