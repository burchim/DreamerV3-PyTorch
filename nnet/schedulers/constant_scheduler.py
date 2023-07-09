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

# Neural Nets
from nnet.schedulers import Scheduler

class ConstantScheduler(Scheduler):

    def __init__(self, val):
        super(ConstantScheduler, self).__init__()

        # Scheduler Params
        self.val = val

    def get_val_step(self, step):
        return self.val