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

# DeepMind Control Envs
from .deep_mind_control_env import DeepMindControlEnv
from .cheetah import Cheetah
from .walker import Walker
from .hopper import Hopper
from .humanoid import Humanoid
from .pendulum import Pendulum
from .swimmer import Swimmer
from .fish import Fish
from .cartpole import Cartpole
from .reacher import Reacher
from .manipulator import Manipulator
from .quadruped import Quadruped
from .acrobot import Acrobot
from .finger import Finger
from .ball_in_cup import BallInCup

# DeepMind Control Envs Dictionary
dm_control_dict = {
    "DeepMindControlEnv": DeepMindControlEnv,
    "Cheetah": Cheetah,
    "Walker": Walker,
    "Hopper": Hopper,
    "Humanoid": Humanoid,
    "Pendulum": Pendulum,
    "Swimmer": Swimmer,
    "Fish": Fish,
    "Cartpole": Cartpole,
    "Reacher": Reacher,
    "Manipulator": Manipulator,
    "Quadruped": Quadruped,
    "Acrobot": Acrobot,
    "Finger": Finger,
    "BallInCup": BallInCup,
}