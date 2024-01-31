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

# Neural Nets
from nnet import structs

# Other
import threading

class BatchEnv:

    def __init__(self, envs, parallel=False):

        self.envs = envs
        self.num_actions = envs[0].num_actions
        self.action_repeat = envs[0].action_repeat
        self.parallel = parallel
        self.num_envs = len(self.envs)

        if hasattr(self.envs[0], "state_shape"):
            self.tuple_state = isinstance(self.envs[0].state_shape, tuple)
        else:
            self.tuple_state = False

        if self.tuple_state:
            self.num_states = len(self.envs[0].state_shape)
        else:
            self.num_states = 1
            
        if hasattr(self.envs[0], "clip_low"):
            self.clip_low = self.envs[0].clip_low
        if hasattr(self.envs[0], "clip_high"):
            self.clip_high = self.envs[0].clip_high

        # Start Threads
        if self.parallel:

            # Locks
            self.locks = [[torch.multiprocessing.Lock() for _ in range(self.num_envs)] for _ in range(3)]

            # Acquire Locks
            for env_i in range(self.num_envs):
                self.locks[0][env_i].acquire()
                self.locks[2][env_i].acquire()

            # Start Threads
            for env_i in range(self.num_envs):
                threading.Thread(target=self.step_loop, args=(env_i,)).start()

            # Create Buffer
            self.buffer = structs.AttrDict(
                state=tuple((torch.zeros(self.num_envs, 3, 64, 64, dtype=torch.uint8), torch.zeros(self.num_envs, 1178, dtype=torch.float32))), # minerl
                #state=torch.zeros(self.num_envs, 3, 64, 64, dtype=torch.uint8), # dmc / atari
                reward=torch.zeros(self.num_envs),
                done=torch.zeros(self.num_envs),
                is_first=torch.zeros(self.num_envs),
                is_last=torch.zeros(self.num_envs),
                error=torch.zeros(self.num_envs),
            )

    def step_loop(self, env_i):

        # Init state
        self.locks[1][env_i].acquire()

        # Step loop
        while 1:

            # Acquire / Release Locks
            self.locks[0][env_i].acquire()
            self.locks[1][env_i].release()
            self.locks[2][env_i].acquire()
            
            # Env Step
            obs = self.envs[env_i].step(self.actions[env_i])

            # Update Buffers
            for key, value in obs.items():

                # Append
                if key == "state" and self.tuple_state:
                    for s_i, s in enumerate(obs.state):
                        self.buffer[key][s_i][env_i] = s
                else:
                    self.buffer[key][env_i] = value

            # Acquire / Release Locks
            self.locks[0][env_i].release()
            self.locks[1][env_i].acquire()
            self.locks[2][env_i].release()

    def sample(self):

        actions = []
        for i, env in enumerate(self.envs):
            actions.append(env.sample())

        actions = torch.stack(actions, dim=0)

        return actions
    
    def step(self, actions):

        if self.parallel:
            return self.step_parallel(actions)
        else:
            return self.step_default(actions)

    def step_parallel(self, actions):

        self.actions = actions

        # Acquire / Release Locks
        for env_i in range(self.num_envs):
            self.locks[0][env_i].release()
            self.locks[1][env_i].acquire()
            self.locks[2][env_i].release()

        # Acquire / Release Locks
        for env_i in range(self.num_envs):
            self.locks[0][env_i].acquire()
            self.locks[1][env_i].release()
            self.locks[2][env_i].acquire()
            
        return self.buffer

    def step_default(self, actions):

        # Env step loop
        batch_obs = structs.AttrDict()
        for i, env in enumerate(self.envs):

            # Env Step
            obs = env.step(actions[i])

            for key, value in obs.items():

                # Init
                if key not in batch_obs:
                
                    # Tuple state
                    if key == "state" and self.tuple_state:
                        batch_obs[key] = tuple([] for _ in range(self.num_states))
                    else:
                        batch_obs[key] = []

                # Append
                if key == "state" and self.tuple_state:
                    for s_i, s in enumerate(obs.state):
                        batch_obs[key][s_i].append(s)
                else:
                    batch_obs[key].append(value)

        # Stack Batch
        for key, value in batch_obs.items():
            if key == "state" and self.tuple_state:
                batch_obs[key] = tuple(torch.stack(s, dim=0) for s in batch_obs[key])
            else:
                batch_obs[key] = torch.stack(batch_obs[key], dim=0)

        return batch_obs
    
    def reset(self, env_i=None):

        # Reset Single env
        if env_i is not None:
            return self.envs[env_i].reset()

        # Env step loop
        batch_obs = structs.AttrDict()
        for i, env in enumerate(self.envs):

            # Reset
            obs = env.reset()

            for key, value in obs.items():

                # Init
                if key not in batch_obs:
                
                    # Tuple state
                    if key == "state" and self.tuple_state:
                        batch_obs[key] = tuple([] for _ in range(self.num_states))
                    else:
                        batch_obs[key] = []

                # Append
                if key == "state" and self.tuple_state:
                    for s_i, s in enumerate(obs.state):
                        batch_obs[key][s_i].append(s)
                else:
                    batch_obs[key].append(value)

        # Stack Batch
        for key, value in batch_obs.items():
            if key == "state" and self.tuple_state:
                batch_obs[key] = tuple(torch.stack(s, dim=0) for s in batch_obs[key])
            else:
                batch_obs[key] = torch.stack(batch_obs[key], dim=0)

        return batch_obs

