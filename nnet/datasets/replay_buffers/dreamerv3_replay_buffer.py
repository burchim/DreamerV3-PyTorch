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
import glob

class DreamerV3ReplayBuffer(datasets.Dataset):

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
            save_trajectories=True, 
            collate_fn=utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}], targets_params=[]), 
            buffer_name="DreamerV3ReplayBuffer"
        ):
        super(DreamerV3ReplayBuffer, self).__init__(num_workers=0, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=root)

        # Params
        self.buffer_name = buffer_name # name of buffer directory
        self.buffer_capacity = buffer_capacity # maximum number of steps in ram
        self.epoch_length = epoch_length # dataset epoch length
        self.sample_length = sample_length # L, sample temporal length to return in steps
        self.ram_buffer = collections.OrderedDict()
        self.streams = collections.OrderedDict()
        self.traj_index = torch.tensor(0) # Trajectory unique id
        self.num_steps = torch.tensor(0) # Number of trajectories in ram buffer, must be <= buffer_capacity
        self.last_save_traj_index = 0 # Last Save infos, (epoch necessary to load/save buffer)
        self.worker_id = None # dataloading id
        self.buffer_dir = os.path.join(root, self.buffer_name) # Buffer Dir
        self.save_trajectories = save_trajectories # Save Replay Buffer Trajectories to main memory for checkpointing, also load trajectories from main memory instead of using ram buffer

        # Create Buffer Dir
        if self.save_trajectories and not os.path.isdir(self.buffer_dir):
            os.makedirs(self.buffer_dir, exist_ok=True)

        # Distributed
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def state_dict(self):
        return { 
            "traj_index": self.traj_index,
            "num_steps": self.num_steps,
            "buffer_keys": list(self.ram_buffer.keys())
        }
    
    def load_state_dict(self, state_dict):
        self.traj_index.fill_(state_dict.pop("traj_index"))
        self.num_steps.fill_(state_dict.pop("num_steps"))
        self.load(state_dict.pop("buffer_keys"))

    def save(self):

        # Save Trajs
        if self.save_trajectories:

            # Select Trajs
            save_trajs = {traj_id:self.ram_buffer[traj_id] for traj_id in range(self.last_save_traj_index, self.traj_index)}

            # Save Trajs
            torch.save(save_trajs, os.path.join(self.buffer_dir, "{}_rank{}.torch".format(self.traj_index, self.rank)))

            # Update 
            self.last_save_traj_index = self.traj_index.item()

    def load(self, buffer_keys):
        
        # All Saves
        for path_trajs in glob.glob(os.path.join(self.buffer_dir, "*_rank{}.torch".format(self.rank))):

            # Load Save
            load_trajs = torch.load(path_trajs)

            # Add required trajs
            for key, value in load_trajs.items():
                if key in buffer_keys:
                    self.ram_buffer[key] = value

        # Assert all keys loaded
        assert sorted(buffer_keys) == sorted(list(self.ram_buffer.keys())), "some buffer traj keys are missing, buffer save may be corrupted: {} buffer keys, {} loaded keys".format(len(buffer_keys), len(list(self.ram_buffer.keys())))

        # Update 
        self.last_save_traj_index = self.traj_index.item()

    def enforce_capacity(self):

        # Pop episodes
        while self.num_steps > self.buffer_capacity:

            # Pop oldest Episode
            oldest_episode_id = (self.traj_index - self.num_steps).item()
            self.ram_buffer.pop(oldest_episode_id)

            # Update Number of steps
            self.num_steps -= 1

    def append_step(self, sample, sample_id):

        # None sample
        if sample is None:
            return

        # Init Stream
        if sample_id not in self.streams:
            self.streams[sample_id] = []

        # Select stream
        stream = self.streams[sample_id]

        # Update Stream
        stream.append([s.clone() for s in sample]) # Clone

        # Unfinished Trajectory
        if len(stream) < self.sample_length:
            return self.state_dict()
        assert len(stream) == self.sample_length

        # Order Traj (elt, time) without memory copy
        traj = [[stream[t][elt] for t in range(self.sample_length)] for elt in range(len(stream[0]))]

        # Slice Stream first element
        self.streams[sample_id].pop(0)

        # Add to ram buffer (using tensor instead of int as key will replace instead of adding)
        self.ram_buffer[self.traj_index.item()] = traj

        # Update Index
        self.traj_index += 1
        self.num_steps += 1

        # enforse num_steps <= buffer_capacity
        self.enforce_capacity()

        return self.state_dict()

    def __len__(self):

        return self.epoch_length * self.batch_size

    def __getitem__(self, n):

        # Sample
        sample = self.sample()

        return sample
    
    def sample(self):

        
        # Select Episode from ram
        traj = self.ram_buffer[random.choice(list(self.ram_buffer.keys()))]

        traj = [torch.stack(elt, axis=0) for elt in traj]

        return traj