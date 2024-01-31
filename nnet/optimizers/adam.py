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
import torch.optim as optim

# NeuralNets
from nnet import schedulers

# Other
from typing import Generator

class Adam(optim.Adam):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, grad_clip_value=None, grad_max_norm=None):

        # Assert types
        assert isinstance(params, Generator) or isinstance(params, list)
        assert isinstance(lr, float) or isinstance(lr, list) or isinstance(lr, schedulers.Scheduler)

        # Convert Params to Param Groups (code from Optimizer)
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        
        # Learning Rate Schedulers
        for group in param_groups:

            # Set Default
            group.setdefault("lr", lr)
            group.setdefault("grad_clip_value", grad_clip_value)
            group.setdefault("grad_max_norm", grad_max_norm)

            # Set lr scheduler
            if isinstance(group["lr"], schedulers.Scheduler):
                group["lr_scheduler"] = group["lr"]
            else:
                group["lr_scheduler"] = schedulers.ConstantScheduler(val=group["lr"])

        # Init Optimizer
        super(Adam, self).__init__(params=param_groups, lr=torch.tensor(0), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def step(self, closure=None):

        # Iter Groups
        for i, group in enumerate(self.param_groups):

            # Update lr
            group['lr'] = group['lr_scheduler'].step()

            # To Float
            if isinstance(group['lr'], torch.Tensor):
                group['lr'] = group['lr'].item()

            # Norm Grads
            if group['grad_max_norm'] is not None:
                group["grad_norm"] = torch.nn.utils.clip_grad_norm_(group["params"], group['grad_max_norm'])

            # Clip Grads
            if group['grad_clip_value'] is not None:
                torch.nn.utils.clip_grad_value_(group["params"], group['grad_clip_value'])

        # Optim step
        return super(Adam, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(Adam, self).state_dict()

        # Append Scheduler Step
        state_dict["model_step"] = self.param_groups[0]["lr_scheduler"].model_step

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step
        for i, group in enumerate(self.param_groups):
            group["lr_scheduler"].model_step.fill_(state_dict["param_groups"][i]["lr_scheduler"].model_step)
            state_dict["param_groups"][i]["lr_scheduler"].model_step = group["lr_scheduler"].model_step
        
        # Load State Dict
        super(Adam, self).load_state_dict(state_dict)

        # Load Scheduler Step
        for i, group in enumerate(self.param_groups):
            group["lr_scheduler"].model_step = state_dict["param_groups"][i]["lr_scheduler"].model_step