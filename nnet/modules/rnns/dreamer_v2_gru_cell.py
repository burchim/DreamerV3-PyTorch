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
from torch import nn

# NeuralNets
from nnet import inits
from nnet import modules

class DreamerV2GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, weight_init="xavier_uniform", bias_init="zeros"):
        super(DreamerV2GRUCell, self).__init__()

        # Params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.update_bias = -1

        # Weights and biases
        self.linear = modules.Linear(input_size + hidden_size, 3 * hidden_size, weight_init=weight_init, bias=bias_init)
        self.layernorm = nn.LayerNorm(3 * hidden_size, eps=1e-3)

    def forward(self, x, hidden):

        # Forward
        h = self.forward_rnn(x, hidden)

        # New Hidden
        new_hidden = h

        return h, new_hidden

    def forward_rnn(self, input, state):

        # Linear Proj
        parts = self.linear(torch.cat([input, state], dim=-1))

        # Norm
        dtype = parts.dtype
        parts = parts.type(torch.float32)
        parts = self.layernorm(parts)
        parts = parts.type(dtype)

        # Chunk
        reset, cand, update = parts.chunk(chunks=3, dim=-1)

        # Apply reset Sigmoid
        reset = torch.sigmoid(reset)

        # Apply cand Tanh
        cand = torch.tanh(reset * cand)

        # Apply update Sigmoid
        update = torch.sigmoid(update + self.update_bias)

        # Hidden
        h = update * cand + (1 - update) * state

        return h