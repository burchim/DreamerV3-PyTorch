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
import torch.nn as nn

class PermuteChannels(nn.Module):

    """ Permute Channels

    Channels_last to channels_first / channels_first to channels_last
    
    """

    def __init__(self, to_last=True, num_dims=None, make_contiguous=False):
        super(PermuteChannels, self).__init__()

        # To last
        self.to_last = to_last

        # Set dims
        if num_dims != None:
            self.set_dims(num_dims)
        else:
            self.dims = None

        # Make Contiguous
        self.make_contiguous = make_contiguous

    def set_dims(self, num_dims):

        if self.to_last:
            self.dims = (0,) + tuple(range(2, num_dims + 2)) + (1,)
        else:
            self.dims = (0, num_dims + 1) + tuple(range(1, num_dims + 1))

    def forward(self, x):

        if self.dims == None:
            self.set_dims(num_dims=x.dim()-2)

        x = x.permute(self.dims)

        if self.make_contiguous:
            x = x.contiguous()

        return x