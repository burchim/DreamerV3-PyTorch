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
from nnet import modules

class DreamerV3GRUCell(nn.Module):

    def __init__(
            self, 
            input_size, 
            hidden_size, 
            weight_init="dreamerv3_normal",
            norm={"class": "LayerNorm", "params": {"eps": 1e-3}}
        ):
        super(DreamerV3GRUCell, self).__init__()

        # Params
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weights and biases
        self.linear = modules.Linear(input_size + hidden_size, 3 * hidden_size, weight_init=weight_init, bias=False)
        self.norm = modules.norm_dict[norm["class"]](3 * hidden_size, **norm["params"], channels_last=True)

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
        parts = self.norm(parts)

        # Chunk
        reset, cand, update = parts.chunk(chunks=3, dim=-1)

        # Apply reset Sigmoid
        reset = torch.sigmoid(reset)

        # Apply cand Tanh
        cand = torch.tanh(reset * cand)

        # Apply update Sigmoid
        update = torch.sigmoid(update - 1)

        # Hidden
        h = update * cand + (1 - update) * state

        return h