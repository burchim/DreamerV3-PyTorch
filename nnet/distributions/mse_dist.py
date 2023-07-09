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

class MSEDist:
    
    def __init__(self, mode, agg="sum", reinterpreted_batch_ndims=0):
        self._mode = mode
        self._agg = agg
        self.reduce_dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])

    def mode(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(dim=self.reduce_dims)
        elif self._agg == "sum":
            loss = distance.sum(dim=self.reduce_dims)
        else:
            raise NotImplementedError(self._agg)
        return - loss