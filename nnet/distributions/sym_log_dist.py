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

# NeuralNets
from nnet import modules

class SymLogDist:

    def __init__(self, mode, reinterpreted_batch_ndims, eps=1e-8):
        self._mode = mode
        self.dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])
        self.eps = eps

    def mode(self):
        return modules.sym_exp(self._mode)
    
    def mean(self):
        return modules.sym_exp(self._mode)
    
    def log_prob(self, value):

        # assert
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        # L2 dist
        distance = (self._mode - modules.sym_log(value)) ** 2

        # eps
        distance = torch.where(distance < self.eps, 0, distance)

        # Reduction
        loss = distance.sum(self.dims)

        return - loss
    
