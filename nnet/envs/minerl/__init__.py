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

# MineRL Envs
try:
    from .minerl_env import MineRLEnv
    from .minecraft_base import MinecraftBase
    from .minecraft_diamond import MinecraftDiamond

    # MineRL Envs Dictionary
    minerl_dict = {
        "MinecraftDiamond": MinecraftDiamond
    }
except Exception as e:
    print(e)
    minerl_dict = {}