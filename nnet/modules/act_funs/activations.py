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
import torch.nn as nn

###############################################################################
# Activation Functions
###############################################################################

class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class ReLU(nn.ReLU):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace=inplace)

    def forward(self, x):
        return super(ReLU, self).forward(x)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()

class TanhGLU(nn.Module):
    
    def __init__(self, dim):
        super(TanhGLU, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in.tanh() * x_gate.sigmoid()
    
class Sigmoid2(nn.Module):

    def __init__(self):
        super(Sigmoid2, self).__init__()
    
    def forward(self, x):
        return 2.0 * torch.sigmoid(x / 2.0)
    
class SymLog(nn.Module):

    def __init__(self):
        super(SymLog, self).__init__()
    
    def forward(self, x):
        return sym_log(x)
    
def sym_log(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
class SymExp(nn.Module):

    def __init__(self):
        super(SymExp, self).__init__()
    
    def forward(self, x):
        return sym_exp(x)
    
def sym_exp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

###############################################################################
# Activation Function Dictionary
###############################################################################

act_dict = {
    None: Identity,
    "Identity": Identity,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Tanh": nn.Tanh,
    "ReLU": ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GLU": nn.GLU,
    "Swish": Swish,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "Sigmoid2": Sigmoid2,
    "SymLog": SymLog,
    "SymExp": SymExp
}