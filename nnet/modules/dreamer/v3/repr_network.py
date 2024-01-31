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

class ReprNetwork(nn.Module):

    def __init__(
        self, 
        dim_input_cnn=3, 
        dim_cnn=48, 
        act_fun=nn.SiLU, 
        weight_init="dreamerv3_normal", 
        bias_init="zeros", 
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
        dim_input_mlp=1178, 
        num_mlp_layers=5, 
        hidden_size=1024, 
        mlp_norm={"class": "LayerNorm", "params": {"eps": 1e-3}}
    ):
        super(ReprNetwork, self).__init__()

        self.dim_cnn = dim_cnn

        # 64 -> 32 -> 16 -> 8 -> 4
        self.cnn = modules.ConvNeuralNetwork(  
            dim_input=dim_input_cnn,
            dim_layers=[dim_cnn, 2*dim_cnn, 4*dim_cnn, 8*dim_cnn],
            kernel_size=4,
            strides=2,
            act_fun=act_fun,
            padding="same",
            weight_init=weight_init, 
            bias_init=bias_init,
            norm=cnn_norm,
            channels_last=False,
            bias=cnn_norm is None
        )

        if dim_input_mlp is not None:
            self.mlp = modules.MultiLayerPerceptron(
                dim_input=dim_input_mlp, 
                dim_layers=[hidden_size for _ in range(num_mlp_layers)], 
                act_fun=act_fun, 
                weight_init=weight_init, 
                bias_init=bias_init, 
                norm=mlp_norm,
                bias=mlp_norm is None
            )

    def forward_cnn(self, x):

        shape = x.shape

        # (B, L, C, H, W) -> (B*L, C, H, W) / (B, C, H, W) -> (B, C, H, W)
        x = x.reshape((-1,) + shape[-3:])

        # (N, C, 64, 64) -> (N, C, 4, 4)
        x = self.cnn(x)

        # (N, C, 4, 4) -> (B, L, 4*4 * C) / (N, C, 4, 4) -> (N, 4*4 * C)
        x = x.reshape(shape[:-3] + (4*4 * 8*self.dim_cnn,))

        return x
    
    def forward_mlp(self, x):

        # Apply SymLog
        x = modules.sym_log(x)

        # MLP Layers
        x = self.mlp(x)

        return x

    def forward(self, inputs):

        # To list
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        # Parse inputs
        inputs_cnn = []
        inputs_mlp = []
        for input in inputs:
            if input.dim() >= 4:
                inputs_cnn.append(input)
            else:
                inputs_mlp.append(input)

        # Forward
        outputs = []
        if len(inputs_cnn) > 0:
            outputs.append(self.forward_cnn(torch.cat(inputs_cnn, dim=-1)))
        if len(inputs_mlp) > 0:
            outputs.append(self.forward_mlp(torch.cat(inputs_mlp, dim=-1)))
        
        # Concat outputs
        outputs = torch.cat(outputs, dim=-1)

        return outputs