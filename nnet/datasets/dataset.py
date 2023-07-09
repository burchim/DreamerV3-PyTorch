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
from nnet import utils

class Dataset(torch.utils.data.Dataset):

    def __init__(self, num_workers=0, batch_size=8, collate_fn=utils.CollateDefault(), root="datasets", shuffle=True, persistent_workers=False):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.root = root
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False