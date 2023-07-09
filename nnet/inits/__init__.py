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

# Inits
from .inits import ones_, zeros_, normal_, uniform_, scaled_uniform_, scaled_normal_, lecun_uniform_, lecun_normal_, he_uniform_, he_normal_, xavier_uniform_, xavier_normal_, normal_02_, orthogonal_, dreamerv3_normal_

# Inits Dictionary
init_dict = {
    "ones": ones_,
    "zeros": zeros_,

    "normal": normal_,
    "uniform": uniform_,

    "scaled_uniform": scaled_uniform_,
    "scaled_normal": scaled_normal_,

    "lecun_uniform": lecun_uniform_,
    "lecun_normal": lecun_normal_,

    "he_uniform": he_uniform_,
    "he_normal": he_normal_,

    "xavier_uniform": xavier_uniform_,
    "xavier_normal": xavier_normal_,

    "normal_02": normal_02_,

    "orthogonal": orthogonal_,

    "dreamerv3_normal": dreamerv3_normal_
}