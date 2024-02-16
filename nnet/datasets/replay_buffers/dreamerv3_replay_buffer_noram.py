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

# Other
import os
import random
import collections

class DreamerV3ReplayBufferNoRAM(datasets.Dataset):

    """ DreamerV3 Replay Buffer

    Save new trajectory for every new step, instead of for every new episode like in DreamerV1 and DreamerV2

    buffer infos:
        traj_index: unique index of trajectory
        num_steps: number of trajectories in buffer, need to be <= buffer capacity
    
    """

    def __init__(
            self,
            batch_size, 
            root, 
            buffer_capacity,
            epoch_length, 
            sample_length,
            shuffle=True,
            collate_fn=utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[]), 
            buffer_name="DreamerV3ReplayBufferNoRAM"
        ):
        super(DreamerV3ReplayBufferNoRAM, self).__init__(num_workers=0, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=root)

        # Buffer Params
        self.buffer_name = buffer_name # name of buffer directory
        self.buffer_capacity = buffer_capacity # maximum number of steps in ram
        self.epoch_length = epoch_length # dataset epoch length
        self.sample_length = sample_length # L, sample temporal length to return in steps
        self.buffer_dir = os.path.join(root, self.buffer_name) # Buffer Dir
        # Buffer State
        self.streams = collections.OrderedDict() # streams of traj steps, <= epoch_length
        self.traj_indices = collections.OrderedDict() # Traj unique indices for each env
        self.num_steps = 0 # Number of trajectories in buffer, must be <= buffer_capacity
        self.traj_keys = [] # List of traj keys
        self.traj_keys_delay = [] # list used to store delayed traj keys in order have a progressively growing buffer

        # Create Buffer Dir
        if not os.path.isdir(self.buffer_dir):
            os.makedirs(self.buffer_dir, exist_ok=True)

        # Distributed
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def state_dict(self):
        return { 
            # "streams": self.streams,
            "traj_indices": self.traj_indices,
            "num_steps": self.num_steps,
            "traj_keys": list(self.traj_keys)
        }
    
    def load_state_dict(self, state_dict):
        #self.streams = state_dict.pop("streams")
        self.traj_indices = state_dict.pop("traj_indices")
        self.num_steps = state_dict.pop("num_steps")

        # Load List keys
        while len(self.traj_keys) > 0:
            self.traj_keys.pop(0)
        self.traj_keys += state_dict.pop("traj_keys")

    def enforce_capacity(self):

        # Pop traj
        while self.num_steps > self.buffer_capacity:

            # Pop oldest Traj key
            self.traj_keys.pop(0)

            # Update Number of steps
            self.num_steps -= 1

    def append_step(self, sample, sample_id):

        # None sample
        if sample is None:
            return

        # Init sample_id structs
        if sample_id not in self.streams:
            self.streams[sample_id] = []
        if sample_id not in self.traj_indices:
            self.traj_indices[sample_id] = 0

        # Select stream
        stream = self.streams[sample_id]

        # Update Stream
        stream.append([s.clone() for s in sample]) # Clone

        # Unfinished Trajectory
        if len(stream) < self.sample_length:
            return self.state_dict()
        assert len(stream) == self.sample_length

        # Create Traj Key
        traj_key = "{}_{}_{}".format(sample_id, self.traj_indices[sample_id], self.rank)

        # Save Trajectory every sample_length, reset ram buffer
        if self.traj_indices[sample_id] % self.sample_length == 0:
            traj = [torch.stack([stream[t][elt] for t in range(self.sample_length)], dim=0) for elt in range(len(stream[0]))]
            torch.save(traj, os.path.join(self.buffer_dir, traj_key + ".torch"))

        # Pop Stream first element
        self.streams[sample_id].pop(0)

        # Append to keys with delay of sample_length-1 steps
        self.traj_keys_delay.append(traj_key)
        traj_key_delay = "{}_{}_{}".format(sample_id, self.traj_indices[sample_id]-self.sample_length+1, self.rank)
        if traj_key_delay in self.traj_keys_delay:
            self.traj_keys.append(self.traj_keys_delay.pop(self.traj_keys_delay.index(traj_key_delay)))
            self.num_steps += 1

        # Update Traj Index
        self.traj_indices[sample_id] += 1

        # enforse num_steps <= buffer_capacity
        self.enforce_capacity()

        return self.state_dict()

    def __len__(self):

        return self.epoch_length * self.batch_size

    # Run in fork process, require share mem
    def __getitem__(self, n):

        # Sample
        sample = self.sample()

        return sample
    
    # Run in fork process, require share mem
    def sample(self):        

        # Select Traj Key
        traj_key = random.choice(self.traj_keys)
        traj_id, traj_index, traj_rank = traj_key.split("_")
        traj_id, traj_index, traj_rank = int(traj_id), int(traj_index), int(traj_rank)

        # remainder samples to save traj
        remainder = traj_index % self.sample_length

        # Load traj from main memory
        if remainder==0:
            traj = torch.load(os.path.join(self.buffer_dir, "{}_{}_{}.torch".format(traj_id, traj_index, traj_rank)))

        # Concat two nearest trajs
        else:
            traj_index_0 = traj_index - remainder
            traj_key_0 = "{}_{}_{}".format(traj_id, traj_index_0, traj_rank)
            traj_0 = torch.load(os.path.join(self.buffer_dir, traj_key_0 + ".torch"))

            traj_index_1 = traj_index_0 + self.sample_length
            traj_key_1 = "{}_{}_{}".format(traj_id, traj_index_1, traj_rank)
            traj_1 = torch.load(os.path.join(self.buffer_dir, traj_key_1 + ".torch"))

            traj = [torch.cat([elt_0[remainder:], elt_1[:remainder]], dim=0) for elt_0, elt_1 in zip(traj_0, traj_1)]

        return traj