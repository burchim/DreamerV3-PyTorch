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
from nnet import distributions

class ObsNetwork(nn.Module):

    def __init__(self, feat_size=32*32+4096, dim_cnn=96, dim_output_cnn=3, act_fun=nn.SiLU, weight_init="dreamerv3_normal", bias_init="zeros", norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, dim_output_mlp=1178, num_mlp_layers=5, hidden_size=1024, mlp_norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, dist_weight_init="xavier_uniform", dist_bias_init="zeros"):
        super(ObsNetwork, self).__init__()

        self.feat_size = feat_size
        self.dim_cnn = dim_cnn
        self.dim_output_mlp = dim_output_mlp

        self.proj = modules.Linear(feat_size, 8 * dim_cnn * 4 * 4, weight_init="xavier_uniform", bias_init="zeros")

        # 4 -> 8 -> 16 -> 32 -> 64
        self.cnn = modules.ConvTransposeNeuralNetwork(  
            dim_input=8*dim_cnn,
            dim_layers=[4*dim_cnn, 2*dim_cnn, 1*dim_cnn, dim_output_cnn],
            kernel_size=[4, 4, 4, 4],
            strides=2,
            act_fun=[act_fun, act_fun, act_fun, None],
            weight_init=[weight_init, weight_init, weight_init, dist_weight_init], 
            bias_init=[bias_init, bias_init, bias_init, dist_bias_init], 
            norm=[norm, norm, norm, None],
            bias=[norm is None, norm is None, norm is None, True],
            padding=1,
            output_padding=0,
            channels_last=False
        )

        if dim_output_mlp is not None:
            self.mlp = modules.MultiLayerPerceptron(dim_input=feat_size, dim_layers=[hidden_size for _ in range(num_mlp_layers)], act_fun=act_fun, weight_init=weight_init, bias_init=bias_init, norm=mlp_norm, bias=mlp_norm is None)
            self.mlp_proj = modules.Linear(hidden_size, dim_output_mlp, weight_init=dist_weight_init, bias_init=dist_bias_init)

    def forward_cnn(self, x):

        # (B, N, D) -> (B, N, C * 4 * 4)
        x = self.proj(x)

        # (B, N, C * 4 * 4) -> (B * N, C, 4 * 4)
        shape = x.shape
        x = x.reshape(-1, 8 * self.dim_cnn, 4, 4)

        # (B * N, C, 4, 4) -> (B * N, 3, 64, 64)
        x = self.cnn(x)

        # Add 0.5 for image scaling [0:1]
        #x = x + 0.5

        # (B * N, 3, 64, 64) -> (B, N, 3, 64, 64)
        x = x.reshape(shape[:-1] + x.shape[1:])

        # Normal Distribution
        obs_dist = distributions.MSEDist(x, reinterpreted_batch_ndims=3)

        return obs_dist
    
    def forward_mlp(self, x):

        x = self.mlp(x)
        x = self.mlp_proj(x)

        obs_dist = distributions.SymLogDist(mode=x, reinterpreted_batch_ndims=1)

        return obs_dist
    
    def forward(self, inputs):

        # Outputs
        outputs = []

        # Forward
        outputs.append(self.forward_cnn(inputs))
        if self.dim_output_mlp is not None:
            outputs.append(self.forward_mlp(inputs))
        
        # To Tensor
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs