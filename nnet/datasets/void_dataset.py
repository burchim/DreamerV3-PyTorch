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
from nnet import datasets
from nnet import utils

class VoidDataset(datasets.Dataset):

    def __init__(self, num_steps=100):
        super(VoidDataset, self).__init__(batch_size=1, collate_fn=utils.CollateFn(inputs_params=[{ "axis": 0 }], targets_params=[{ "axis": 0 }]), shuffle=False, root=None)
        self.num_steps = num_steps

    def __getitem__(self, n):
        return torch.tensor(n),

    def __len__(self):
        return self.num_steps
