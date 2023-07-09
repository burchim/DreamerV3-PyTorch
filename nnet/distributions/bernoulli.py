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

class Bernoulli(torch.distributions.Bernoulli):

    def __init__(self, probs=None, logits=None, validate_args=None):
        super(Bernoulli, self).__init__(probs=probs, logits=logits, validate_args=validate_args)

    def log_prob(self, x):
        logits = self.logits
        log_probs0 = - F.softplus(logits)
        log_probs1 = - F.softplus(-logits)

        return log_probs0 * (1-x) + log_probs1 * x
    
    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode