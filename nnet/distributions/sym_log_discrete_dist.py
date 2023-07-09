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
import torch.nn.functional as F

# NeuralNets
from nnet import modules

class SymLogDiscreteDist:

    def __init__(self, logits, reinterpreted_batch_ndims=1, low=-20, high=20, transform_forward=modules.sym_log, transform_backward=modules.sym_exp):
        self.logits = logits
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.probs = logits.softmax(dim=-1)
        self.reduce_dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])
        self.bins = torch.linspace(low, high, steps=logits.shape[-1], device=logits.device, dtype=logits.dtype)
        self.transform_forward = transform_forward
        self.transform_backward = transform_backward

    def mode(self):
        return self.transform_backward(torch.sum(self.probs * self.bins, dim=-1, keepdim=True))
    
    def mean(self):
        return self.transform_backward(torch.sum(self.probs * self.bins, dim=-1, keepdim=True))
    
    def log_prob(self, x):

        # sym log target (..., 1)
        x = self.transform_forward(x)

        # (..., 1) -> (..., 1, N) -> (..., 1)
        below = torch.sum((self.bins <= x.unsqueeze(dim=-1)).type(torch.int32), dim=-1) - 1
        above = len(self.bins) - torch.sum((self.bins > x.unsqueeze(dim=-1)).type(torch.int32), dim=-1)

        # clip 0:N-1
        below = torch.clip(below, 0, len(self.bins) - 1)
        above = torch.clip(above, 0, len(self.bins) - 1)

        # Equal
        equal = (below == above)

        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        
        # (..., 1)
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # (..., 1) -> # (..., 1, N)
        target = (F.one_hot(below, num_classes=len(self.bins)) * weight_below.unsqueeze(dim=-1) + F.one_hot(above, len(self.bins)) * weight_above.unsqueeze(dim=-1))

        # Normalize
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        
        target = target.squeeze(dim=-2)

        # (..., N) -> (...)
        return (target * log_pred).sum(dim=-1)