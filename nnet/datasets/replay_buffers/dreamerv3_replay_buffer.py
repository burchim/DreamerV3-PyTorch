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
import copy

class DreamerV3ReplayBuffer(datasets.Dataset):

    def __init__(
            self, 
            num_workers, 
            batch_size, 
            root, 
            buffer_capacity,
            epoch_length, 
            sample_length, 
            include_done=False, 
            shuffle=True,
            save_trajectories=True, 
            collate_fn=utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}], targets_params=[]), 
            buffer_name="DreamerV3ReplayBuffer"
        ):
        super(DreamerV3ReplayBuffer, self).__init__(num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=root)

        # to fix num_workers
        assert not (num_workers > 0 and not save_trajectories), "not implemented"

        # Params
        self.buffer_name = buffer_name # name of buffer directory
        self.buffer_capacity = buffer_capacity # maximum number of steps in ram
        self.epoch_length = epoch_length # dataset epoch length
        self.sample_length = sample_length # L, sample temporal length to return in steps
        self.include_done = include_done # include done when loading episode
        self.ram_buffer = collections.OrderedDict() # Init Buffer as Ordered dict
        self.streams = collections.OrderedDict()
        self.num_steps = torch.tensor(0) # Num steps in ram buffer, must be <= buffer_capacity
        self.traj_index = torch.tensor(0) # Trajectory unique id
        self.num_trajs = torch.tensor(0) # Number of trajectories in ram buffer
        self.worker_id = None # dataloading id
        self.buffer_dir = os.path.join(root, self.buffer_name) # Buffer Dir
        self.save_trajectories = save_trajectories # Save Replay Buffer Episodes

        # Create Buffer Dir
        if self.save_trajectories and not os.path.isdir(self.buffer_dir):
            os.makedirs(self.buffer_dir, exist_ok=True)

        # Multi Workers
        if self.num_workers > 0:

            # Moves the underlying storage to shared memory
            self.num_steps.share_memory_()
            self.traj_index.share_memory_()
            self.num_trajs.share_memory_()

            # Persistent workers after each epoch
            self.persistent_workers = True

        # Distributed
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def get_buffer_state(self):
        return {
            "buffer_steps": self.num_steps, 
            "buffer_traj_id": self.traj_index
        }

    def restore_buffer(self):

        pass

    def enforce_capacity(self):

        # Pop episodes
        while self.num_steps > self.buffer_capacity:

            # Pop oldest Episode
            if not self.save_trajectories:
                oldest_traj_id, oldest_traj = self.ram_buffer.popitem(last=False)

            # Update Number of steps
            self.num_steps -= self.sample_length
            self.num_trajs -= 1

    def append_step(self, sample, sample_id):

        # None sample
        if sample is None:
            return
        
        # Copy sample to avoid memory modif
        sample = copy.deepcopy(sample)

        # Init Stream
        if sample_id not in self.streams:
            self.streams[sample_id] = []

        # Select stream
        stream = self.streams[sample_id]

        # Update Stream
        stream.append(sample)

        # Unfinished Trajectory
        if len(stream) < self.sample_length:
            return self.get_buffer_state()
        assert len(stream) == self.sample_length
        
        # Stack Trajectory
        traj = [torch.stack([stream[t][elt] for t in range(self.sample_length)], dim=0) for elt in range(len(stream[0]))]

        # Clear Stream
        self.streams[sample_id] = []

        # Save Sample
        if self.save_trajectories:
            torch.save(traj, os.path.join(self.buffer_dir, "{}_rank{}.torch".format(self.traj_index, self.rank)))

        # Add to ram buffer (using tensor instead of int as key will replace instead of adding)
        else:
            self.ram_buffer[self.traj_index.item()] = traj

        # Update Index
        self.traj_index += 1
        self.num_trajs += 1

        # Update Number of steps
        self.num_steps += self.sample_length

        # enforse num_steps <= buffer_capacity
        self.enforce_capacity()

        return self.get_buffer_state()

    def __len__(self):

        return self.epoch_length * self.batch_size

    def __getitem__(self, n):

        # Sample
        sample = self.sample()

        return sample
    
    def sample(self):

        
        # Select Episode from main memory
        if self.save_trajectories:
            traj = torch.load(os.path.join(self.buffer_dir, "{}_rank{}.torch".format(random.randint(self.traj_index - self.num_trajs, self.traj_index-1), self.rank)))
        
        # Select Episode from ram
        else:
            traj = random.choice(list(self.ram_buffer.values()))
            
        #print(self.traj_index, self.num_trajs, self.num_steps)

        return traj

