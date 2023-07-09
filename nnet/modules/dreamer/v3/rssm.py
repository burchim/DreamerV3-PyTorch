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
import torch.nn.functional as F

# NeuralNets
from nnet import modules
from nnet import distributions
from nnet import structs

class RSSM(nn.Module):

    def __init__(
            self, 
            num_actions, 
            deter_size=4096, 
            stoch_size=32, 
            hidden_size=1024, 
            act_fun=nn.SiLU, 
            std_fun=modules.Sigmoid2, 
            embed_size=4*4*8*96, 
            discrete=32, 
            min_std=0.1, 
            learn_initial=True, 
            weight_init="dreamerv3_normal", 
            bias_init="zeros", 
            norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
            uniform_mix=0.01, 
            action_clip=1.0, 
            dist_weight_init="xavier_uniform", 
            dist_bias_init="zeros"
        ):
        super(RSSM, self).__init__()

        # Params
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.discrete = discrete
        self.min_std = min_std
        self.std_fun = std_fun()
        self.uniform_mix = uniform_mix
        self.learn_initial = learn_initial
        self.action_clip = action_clip

        self.gru = modules.rnns.DreamerV2GRUCell(hidden_size, deter_size, weight_init=weight_init, bias_init=bias_init)
        if self.discrete:
            self.mlp_img1 = modules.MultiLayerPerceptron(stoch_size * discrete + num_actions, hidden_size, act_fun=act_fun, weight_init=weight_init, bias_init=bias_init, norm=norm, bias=norm is None)
            self.mlp_img2 = modules.MultiLayerPerceptron(deter_size, [hidden_size, self.discrete * stoch_size], act_fun=[act_fun, None], weight_init=[weight_init, dist_weight_init], bias_init=[bias_init, dist_bias_init], norm=[norm, None], bias=[norm is None, True])
            self.mlp_obs1 = modules.MultiLayerPerceptron(embed_size + deter_size, [hidden_size, self.discrete * stoch_size], act_fun=[act_fun, None], weight_init=[weight_init, dist_weight_init], bias_init=[bias_init, dist_bias_init], norm=[norm, None], bias=[norm is None, True])
        else:
            self.mlp_img1 = modules.MultiLayerPerceptron(stoch_size + num_actions, hidden_size, act_fun=act_fun, weight_init=weight_init, bias_init=bias_init, norm=norm, bias=norm is None)
            self.mlp_img2 = modules.MultiLayerPerceptron(deter_size, [hidden_size, 2 * stoch_size], act_fun=[act_fun, None], weight_init=[weight_init, dist_weight_init], bias_init=[bias_init, dist_bias_init], norm=[norm, None], bias=[norm is None, True])
            self.mlp_obs1 = modules.MultiLayerPerceptron(embed_size + deter_size, [hidden_size, 2 * stoch_size], act_fun=[act_fun, None], weight_init=[weight_init, dist_weight_init], bias_init=[bias_init, dist_bias_init], norm=[norm, None], bias=[norm is None, True])
        
        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.deter_size))

    def get_stoch(self, deter):
        
        # MLP Img 2
        if self.discrete:

            # Linear Logits
            logits = self.mlp_img2(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))

            dist_params = {'logits': logits}
        else:
            mean, std = torch.chunk(self.mlp_img2(deter), chunks=2, dim=-1)

            # Born std to [self.min_std:+inf]
            std = self.std_fun(std) + self.min_std

            # Dist Params
            dist_params = {'mean': mean, 'std': std}
    
        # Get Mode
        stoch = self.get_dist(dist_params).mode()

        return stoch

    def initial(self, batch_size=1, dtype=torch.float32, device="cpu", detach_learned=False):

        if self.discrete:
            initial_state = structs.AttrDict(
                logits=torch.zeros(batch_size, self.stoch_size, self.discrete, dtype=dtype, device=device),
                stoch=torch.zeros(batch_size, self.stoch_size, self.discrete, dtype=dtype, device=device),
                deter=torch.zeros(batch_size, self.deter_size, dtype=dtype, device=device)
            )
        else:
            initial_state = structs.AttrDict(
                mean=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                std=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                stoch=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                deter=torch.zeros(batch_size, self.deter_size, dtype=dtype, device=device)
            )

        # Learned Initial
        if self.learn_initial:
            initial_state.deter = F.tanh(self.weight_init).repeat(batch_size, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter) 

            # Detach Learned
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()

        return initial_state

    def observe(self, embed, prev_actions, is_firsts, prev_state=None):

        # Initial State
        if prev_state is None:
            prev_actions[:, 0] = 0.0
            prev_state = self.initial(batch_size=embed.shape[0], dtype=embed.dtype, device=embed.device)

        # Model Recurrent loop
        if self.discrete:
            posts = {"stoch": [], "deter": [], "logits": []}
            priors = {"stoch": [], "deter": [], "logits": []}
        else:
            posts = {"stoch": [], "deter": [], "mean": [], "std": []}
            priors = {"stoch": [], "deter": [], "mean": [], "std": []}
        for l in range(embed.shape[1]):
                
            # Forward Model
            post, prior = self(prev_state, prev_actions[:, l], embed[:, l], is_firsts[:, l])

            # Update previous state, Teacher Forcing
            prev_state = post

            # Append to Lists
            for key, value in post.items():
                posts[key].append(value)
            for key, value in prior.items():
                priors[key].append(value)

        # Stack Lists
        posts = {k: torch.stack(v, dim=1) for k, v in posts.items()} # (B, L, D)
        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()} # (B, L, D)

        return posts, priors

    def imagine(self, p_net, prev_state, img_steps=1):

        # Policy
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()

        # Current state action
        prev_state["action"] = policy(prev_state)

        # Model Recurrent loop with St, At
        if self.discrete:
            img_states = {"stoch": [prev_state["stoch"]], "deter": [prev_state["deter"]], "logits": [prev_state["logits"]], "action": [prev_state["action"]]}
        else:
            img_states = {"stoch": [prev_state["stoch"]], "deter": [prev_state["deter"]], "mean": [prev_state["mean"]], "std": [prev_state["std"]], "action": [prev_state["action"]]}
        for h in range(img_steps):
                
            # Forward Model
            img_state = self.forward_img(prev_state, prev_state["action"])

            # Current state action
            img_state["action"] = policy(img_state)

            # Update previous state
            prev_state = img_state

            # Append to Lists
            for key, value in img_state.items():
                img_states[key].append(value)

        # Stack Lists
        img_states = {k: torch.stack(v, dim=1) for k, v in img_states.items()} # (B, 1+img_steps, D)

        return img_states

    def get_feat(self, state):

        # Flatten stoch size and discrete size
        if self.discrete:
            stoch = state["stoch"].flatten(start_dim=-2, end_dim=-1)
        else:
            stoch = state["stoch"]

        return torch.cat([stoch, state["deter"]], dim=-1)
    
    def get_dist(self, state):

        if self.discrete:
            return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)
        else:
            return torch.distributions.Independent(distributions.Normal(loc=state['mean'], scale=state['std']), 1)

    def forward_img(self, prev_state, prev_action):

        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_action *= (self.action_clip / torch.clip(torch.abs(prev_action), min=self.action_clip)).detach()

        # Flatten stoch size and discrete size
        if self.discrete:
            stoch = prev_state["stoch"].flatten(start_dim=-2, end_dim=-1)
        else:
            stoch = prev_state["stoch"]

        # MLP Img 1
        x = self.mlp_img1(torch.concat([stoch, prev_action], dim=-1))

        # Recurrent
        x, deter = self.gru(x, prev_state["deter"])

        # MLP Img 2
        if self.discrete:

            # Linear Logits
            logits = self.mlp_img2(x).reshape(x.shape[:-1] + (self.stoch_size, self.discrete))

            dist_params = {'logits': logits}
        else:
            mean, std = torch.chunk(self.mlp_img2(x), chunks=2, dim=-1)

            # Born std to [self.min_std:+inf]
            std = self.std_fun(std) + self.min_std

            # Dist Params
            dist_params = {'mean': mean, 'std': std}
    
        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Return Prior
        return {"stoch": stoch, "deter": deter, **dist_params}

    def forward(self, prev_state, prev_action, embed, is_first):

        assert embed.dim() == 2

        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_action *= (self.action_clip / torch.clip(torch.abs(prev_action), min=self.action_clip)).detach()

        # Resest First States and Actions
        if is_first.any():

            # Unsqueeze is_first (B, 1)
            is_first = is_first.unsqueeze(dim=-1)

            # Reset first Actions
            prev_action *= (1.0 - is_first)

            # Reset first States
            init_state = self.initial(embed.shape[0], dtype=embed.dtype, device=embed.device)
            for key, value in prev_state.items():
                is_first_r = torch.reshape(is_first, is_first.shape + (1,) * (len(value.shape) - len(is_first.shape)))
                prev_state[key] = value * (1.0 - is_first_r) + init_state[key] * is_first_r

        # Forward Img
        prior = self.forward_img(prev_state, prev_action)

        # Concat deter and Emb
        emb_h = torch.concat([embed, prior["deter"]], dim=-1)

        # MLP Obs 1
        if self.discrete:
            dist_params = {'logits': self.mlp_obs1(emb_h).reshape(emb_h.shape[:-1] + (self.stoch_size, self.discrete))}
        else:
            mean, std = torch.chunk(self.mlp_obs1(emb_h), chunks=2, dim=-1)

            # Born std to [self.min_std:+inf]
            std = self.std_fun(std) + self.min_std

            # Dist Params
            dist_params = {'mean': mean, 'std': std}

        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Post
        post = {"stoch": stoch, "deter": prior["deter"], **dist_params}

        # Return post and prior
        return post, prior