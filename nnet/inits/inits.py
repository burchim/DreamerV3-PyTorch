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

import math

# PyTorch
import torch.nn.init as init

###############################################################################
# Initializations
###############################################################################

# 1
def ones_(tensor):
    return init.ones_(tensor)

# 0
def zeros_(tensor):
    return init.zeros_(tensor)

# N(mean, std)
def normal_(tensor, mean=0.0, std=1.0):
    return init.normal_(tensor, mean=mean, std=std)

# U(a, b)
def uniform_(tensor, a=0.0, b=1.0):
    return init.uniform_(tensor, a=a, b=b)

# U(-b, b) where b = sqrt(1/fan_in)
def scaled_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, a=math.sqrt(5), mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def scaled_normal_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(3/fan_in)
def lecun_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def lecun_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(6/fan_in)
def he_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, mode=mode)

# N(0, std**2) where std = sqrt(2/fan_in)
def he_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, mode=mode)

# U(-b, b) where b = sqrt(6/(fan_in + fan_out))
def xavier_uniform_(tensor):
    return init.xavier_uniform_(tensor)

# N(0, std**2) where std = sqrt(2/(fan_in + fan_out))
def xavier_normal_(tensor):
    return init.xavier_normal_(tensor)

# N(0.0, 0.02)
def normal_02_(tensor):
    return init.normal_(tensor, mean=0.0, std=0.02)

# Orthogonal
def orthogonal_(tensor):
    return init.orthogonal_(tensor)

# DreamerV3 normal
def dreamerv3_normal_(tensor):

    # Gain
    gain = 1 / 0.87962566103423978

    # Compute std
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    # 2.0 * std truncated normal init
    return init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)