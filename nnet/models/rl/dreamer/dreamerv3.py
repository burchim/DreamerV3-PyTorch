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
from torch import nn
import torchvision

# NeuralNets
from nnet import models
from nnet import optimizers
from nnet import utils
from nnet import envs
from nnet.modules.dreamer import v3 as dreamer_networks
from nnet.structs import AttrDict

# Other
import copy
import itertools
import time
import threading

class DreamerV3(models.Model):

    """ Dreamer V3

    Mastering Diverse Domains through World Models
    https://arxiv.org/abs/2301.04104

    """

    def __init__(self, env_name, override_config={}, name="Dreamer3: Mastering Diverse Domains through World Models"):
        super(DreamerV3, self).__init__(name=name)

        # dreamerv3 Acrobot Swingup
        # Optimizer model_opt has 15,685,251 variables.
        # Optimizer actor_opt has 1,051,650 variables.
        # Optimizer critic_opt has 1,181,439 variables.

        # atari100k breakout
        # Optimizer model_opt has 15,686,787 variables.
        # Optimizer actor_opt has 1,052,676 variables.
        # Optimizer critic_opt has 1,181,439 variables.

        # minecraft
        # Optimizer model_opt has 198,692,381 variables.
        # Optimizer actor_opt has 9,473,049 variables.
        # Optimizer critic_opt has 9,708,799 variables.

        # Model Sizes
        model_sizes = {
            # 8M
            "XS": AttrDict({
                "deter_size": 256,
                "dim_cnn": 24,
                "hidden_size": 256,
                "num_layers": 1
            }),
            # 18M
            "S": AttrDict({
                "deter_size": 512,
                "dim_cnn": 32,
                "hidden_size": 512,
                "num_layers": 2
            }),
            # 37M
            "M": AttrDict({
                "deter_size": 1024,
                "dim_cnn": 48,
                "hidden_size": 640,
                "num_layers": 3
            }),
            # 77M
            "L": AttrDict({
                "deter_size": 2048,
                "dim_cnn": 64,
                "hidden_size": 768,
                "num_layers": 4
            }),
            # 200M
            "XL": AttrDict({
                "deter_size": 4096,
                "dim_cnn": 96,
                "hidden_size": 1024,
                "num_layers": 5
            }),
        }

        # Env Type
        env_name = env_name.split("-")
        self.env_type = env_name[0]
        assert self.env_type in ["dmc", "atari", "atari100k", "minerl", "memory_maze"]

        # Config
        self.config = AttrDict()
        # Env
        if self.env_type == "dmc":
            self.config.env_class = envs.dm_control.dm_control_dict[env_name[1]]
            self.config.env_params = {"task": env_name[2], "mode": "rgb", "history_frames": 1, "img_size": (64, 64), "action_repeat": 2}
            self.config.model_size = "S"
            self.config.time_limit = 1000
        elif self.env_type == "atari100k":
            self.config.env_class = envs.atari.AtariEnv
            self.config.env_params = {"game": env_name[1], "history_frames": 1, "img_size": (64, 64), "action_repeat": 4, "grayscale_obs": False, "noop_max": 30, "repeat_action_probability": 0.0, "full_action_space": False}
            self.config.model_size = "S"
            self.config.time_limit = 108000
        elif self.env_type == "atari":
            self.config.env_class = envs.atari.AtariEnv
            self.config.env_params = {"game": env_name[1], "history_frames": 1, "img_size": (64, 64), "action_repeat": 4, "grayscale_obs": False, "noop_max": 0, "full_action_space": True}
            self.config.model_size = "XL"
            self.config.time_limit = 108000
        elif self.env_type == "minerl":
            self.config.env_class = envs.minerl.MinecraftDiamond
            self.config.env_params = {"img_size": (64, 64)}
            self.config.model_size = "XL"
            self.config.time_limit = 36000
        elif self.env_type == "memory_maze":
            self.config.env_class = envs.memory_maze.MemoryMaze
            self.config.env_params = {"maze_size": env_name[1]}
            self.config.model_size = "XL"
            self.config.time_limit = {"9x9": 1000, "11x11": 2000, "13x13": 3000, "15x15": 4000}[env_name[1]]
        # Training
        self.config.batch_size = 16
        self.config.L = 64
        self.config.H = 15
        self.config.num_envs = {"dmc": 4, "atari100k": 1, "atari": 8, "minerl": 16, "memory_maze": 8}[self.env_type]
        self.config.collate_fn = {
            "dmc": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}], targets_params=[]), 
            "atari100k": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}], targets_params=[]), 
            "atari": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}], targets_params=[]), 
            "minerl": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[]),
            "memory_maze": utils.CollateFn(inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}, {"axis": 5}], targets_params=[])
        }[self.env_type]
        self.config.epochs = {"dmc": 50, "atari100k": 50, "atari": 250, "minerl": 250, "memory_maze": 250}[self.env_type]
        self.config.epoch_length = {"dmc": 5000, "atari100k": 2000, "atari": 12500, "minerl": 6250, "memory_maze": 12500}[self.env_type]
        self.config.precision = {"dmc": torch.float16, "atari100k": torch.float32, "atari": torch.float16, "minerl": torch.float16, "memory_maze": torch.float16}[self.env_type]
        self.config.grad_init_scale = 32.0
        self.config.augments_train = nn.Identity()
        # Eval
        self.config.eval_policy_mode = "sample"
        self.config.eval_epidodes = {"dmc": 10, "atari100k": 10, "atari": 10, "minerl": 0, "memory_maze": 10}[self.env_type]  
        self.config.augments_eval = nn.Identity()      
        # Optimizer
        self.config.opt_weight_decay = 0.0
        self.config.model_lr = 1e-4
        self.config.value_lr = 3e-5
        self.config.actor_lr = 3e-5
        self.config.model_eps = 1e-8
        self.config.value_eps = 1e-5
        self.config.actor_eps = 1e-5
        self.config.model_grad_max_norm = 1000
        self.config.value_grad_max_norm = 100
        self.config.actor_grad_max_norm = 100
        # Env step params
        self.config.correct_train_ratio = True # multiply env_step_period by num_envs factor
        self.config.env_step_period = {"dmc": 512, "atari100k": 1024, "atari": 64, "minerl": 16, "memory_maze": 64}[self.env_type]
        self.config.num_env_steps = 1
        self.config.noise = None
        self.config.parallel_envs = {"dmc": False, "atari100k": False, "atari": False, "minerl": True, "memory_maze": False}[self.env_type] # env step in parallel for num_envs
        self.config.thread_env_step = {"dmc": False, "atari100k": False, "atari": False, "minerl": True, "memory_maze": False}[self.env_type] # env step in other thread than train_step
        # Replay Buffer
        self.config.buffer_capacity = int(1e6)
        self.config.pre_fill_steps = {"dmc": 100, "atari100k": 100, "atari": 100, "minerl": 100, "memory_maze": 100}[self.env_type] # pre_fill_steps in number of buffer samples
        self.config.include_done_transition = {"dmc": False, "atari100k": True, "atari": True, "minerl": True, "memory_maze": False}[self.env_type]
        self.config.prioritize_ends = {"dmc": False, "atari100k": True, "atari": True, "minerl": True, "memory_maze": False}[self.env_type]
        self.config.load_replay_buffer_state_dict = True # Load ReplayBuffer saved state dict
        # Return Norm
        self.config.return_norm_decay = 0.99
        self.config.return_norm_limit = 1.0
        self.config.return_norm_perc_low = 0.05
        self.config.return_norm_perc_high = 0.95
        # Env Params
        self.config.eval_env_params = {}
        self.config.train_env_params = {}
        # Encoder / RSSM / Decoder Params
        model_params = model_sizes[self.config.model_size]
        self.config.norm = {"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}}
        self.config.free_nats = 1.0
        self.config.image_channels = 3
        self.config.dim_cnn = model_params.dim_cnn
        self.config.repr_layers = model_params.num_layers
        self.config.repr_hidden_size = model_params.hidden_size
        self.config.model_discrete = 32
        self.config.model_stoch_size = 32
        self.config.model_deter_size = model_params.deter_size
        self.config.model_hidden_size = model_params.hidden_size
        self.config.action_hidden_size = model_params.hidden_size
        self.config.value_hidden_size = model_params.hidden_size
        self.config.reward_hidden_size = model_params.hidden_size
        self.config.discount_hidden_size = model_params.hidden_size
        self.config.action_layers = model_params.num_layers
        self.config.value_layers = model_params.num_layers
        self.config.reward_layers = model_params.num_layers
        self.config.discount_layers = model_params.num_layers
        self.config.dim_input_mlp = {"dmc": None, "atari100k": None, "atari": None, "minerl": 1178, "memory_maze": None}[self.env_type]
        self.config.dim_output_mlp = {"dmc": None, "atari100k": None, "atari": None, "minerl": 1178, "memory_maze": None}[self.env_type]
        self.config.learn_initial = True
        # Actor Params
        self.config.actor_grad = {"dmc": "dynamics", "atari100k": "reinforce", "atari": "reinforce", "minerl": "reinforce", "memory_maze": "reinforce"}[self.env_type]
        self.config.policy_discrete = {"dmc": False, "atari100k": True, "atari": True, "minerl": True, "memory_maze": True}[self.env_type]
        self.config.eta_entropy = 0.0003
        # Critic Params
        self.config.lambda_td = 0.95
        self.config.gamma = 0.997
        self.config.target_value_reg = True
        self.config.critic_ema_decay = 0.02
        self.config.critic_slow_reg_scale = 1.0
        # Loss Scales
        self.config.loss_reward_scale = 1.0
        self.config.loss_discount_scale = 1.0
        self.config.loss_decoder_scale = 1.0
        self.config.loss_kl_prior_scale = 0.5
        self.config.loss_kl_post_scale = 0.1
        # Log
        self.config.debug = 0
        self.config.log_figure_batch = 16
        self.config.log_figure_context_frames = 5
        self.config.running_rewards_momentum = 0.05

        # Override Config
        for key, value in override_config.items():
            assert key in self.config
            self.config[key] = value

        # Create Envs
        self.env = envs.wrappers.BatchEnv([
            envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.train_env_params)), 
                    time_limit=self.config.time_limit
                )
            )
        for _ in range(self.config.num_envs)], parallel=self.config.parallel_envs)
        if self.config.eval_epidodes > 0:
            self.env_eval = envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(self.config.env_params, **self.config.eval_env_params)),
                    time_limit=self.config.time_limit
                )
            )
        else:
            self.env_eval = None

        # Networks
        self.config.embed_size = 4*4*8 * self.config.dim_cnn if self.config.dim_input_mlp is None else 4*4*8*self.config.dim_cnn + self.config.repr_hidden_size
        feat_size = self.config.model_stoch_size * self.config.model_discrete + self.config.model_deter_size if self.config.model_discrete else self.config.model_stoch_size + self.config.model_deter_size
        self.repr_net = dreamer_networks.ReprNetwork(
            dim_input_cnn=self.config.image_channels, 
            dim_cnn=self.config.dim_cnn, 
            dim_input_mlp=self.config.dim_input_mlp, 
            num_mlp_layers=self.config.repr_layers, 
            hidden_size=self.config.repr_hidden_size,
            cnn_norm=self.config.norm,
            mlp_norm=self.config.norm
        )
        self.rssm = dreamer_networks.RSSM(
            num_actions=self.env.num_actions, 
            deter_size=self.config.model_deter_size, 
            stoch_size=self.config.model_stoch_size, 
            hidden_size=self.config.model_hidden_size, 
            embed_size=self.config.embed_size, 
            discrete=self.config.model_discrete, 
            learn_initial=self.config.learn_initial,
            norm=self.config.norm
        )
        self.p_net = dreamer_networks.PolicyNetwork(
            num_actions=self.env.num_actions, 
            hidden_size=self.config.action_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.action_layers, 
            discrete=self.config.policy_discrete,
            norm=self.config.norm
        )
        self.v_net = dreamer_networks.ValueNetwork(
            hidden_size=self.config.value_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.value_layers,
            norm=self.config.norm
        )
        self.r_net = dreamer_networks.RewardNetwork(
            hidden_size=self.config.reward_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.reward_layers,
            norm=self.config.norm
        )
        self.obs_net = dreamer_networks.ObsNetwork(
            dim_output_cnn=self.config.image_channels, 
            feat_size=feat_size, 
            dim_cnn=self.config.dim_cnn, 
            dim_output_mlp=self.config.dim_output_mlp, 
            num_mlp_layers=self.config.repr_layers, 
            hidden_size=self.config.repr_hidden_size,
            norm=self.config.norm,
            mlp_norm=self.config.norm
        )
        self.discount_net = dreamer_networks.DiscountNetwork(
            hidden_size=self.config.discount_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.discount_layers,
            norm=self.config.norm
        )
        
        # Slow Moving Networks
        self.add_frozen("v_target", copy.deepcopy(self.v_net))

        # Env Step Networks, avoid training mode / weight update during env_step
        if self.config.thread_env_step:
            self.add_frozen("repr_net_env", copy.deepcopy(self.repr_net), persistent=False)
            self.add_frozen("rssm_env", copy.deepcopy(self.rssm), persistent=False)
            self.add_frozen("p_net_env", copy.deepcopy(self.p_net), persistent=False)

        # Percentiles
        self.register_buffer("perc_low", torch.tensor(0.0))
        self.register_buffer("perc_high", torch.tensor(0.0))
        
        # Training Infos
        self.register_buffer("episodes", torch.tensor(0))
        self.register_buffer("running_rewards", torch.tensor(0.0))
        self.register_buffer("ep_rewards", torch.zeros(self.config.num_envs), persistent=False)
        self.register_buffer("action_step", torch.tensor(0))

        # World Model
        self.world_model = self.WorldModel(outer=self)

        # Actor Model
        self.actor_model = self.ActorModel(outer=self)

        # Actor Model
        self.value_model = self.ValueModel(outer=self)

    def preprocess_inputs(self, state):

        if self.training:
            augment = self.config.augments_train
        else:
            augment = self.config.augments_eval

        def norm_image(image):
            return image.type(torch.float32) / 255 - 0.5 if image.dtype==torch.uint8 else image
        
        def augment_image(image):
            assert image.dim() in [4, 5]

            # (T, C, H, W)
            if image.dim() == 4:
                return augment(image)
            
            # (B, T, C, H, W)
            else:
                # Augment per sample of batch
                for b in range(image.shape[0]):
                    image[b] = augment(image[b])

                return image

        # Tuple state
        if isinstance(state, tuple):
            state = tuple(augment_image(norm_image(s)) if s.dim() > 3 else s for s in state)
        # List of Inputs
        elif isinstance(state, list):
            state = [augment_image(norm_image(s)) if s.dim() > 3 else s for s in state]
        # State (could be image of lowd)
        else:
            state = augment_image(norm_image(state)) if state.dim() > 3 else state

        return state

    def save(self, path, save_optimizer=True):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "is_distributed": self.is_distributed or self.is_parallel,
            "ema_model_state_dict": None if self.ema_model == None else self.ema_model.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None,
            "replay_buffer_state_dict": self.replay_buffer.state_dict()
            }, path)
        
        # Save Buffer
        self.replay_buffer.save()

        # Print Model state
        if self.rank == 0:
            print("Model saved at step {}".format(self.model_step))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load Model State Dict
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.load_state_dict({key.replace(".module.", "."):value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)
        else:
            self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step
            self.model_step.fill_(checkpoint["model_step"])

        # Load EMA Model State Dict
        if checkpoint["ema_model_state_dict"] is not None:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        # Load replay Buffer State Dict
        if checkpoint.get("replay_buffer_state_dict", False):
            if self.config.load_replay_buffer_state_dict:
                self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
            elif self.rank == 0 and verbose:
                print("load_replay_buffer_state_dict set to False: replay buffer state dict not loaded")

        # Print Model state
        if self.rank == 0 and verbose:
            print("Rank {}: Model loaded at step {}".format(self.rank, self.model_step))

    def set_replay_buffer(self, replay_buffer):

        # Replay Buffer
        self.replay_buffer = replay_buffer

        # Set History
        obs_reset = self.env.reset()
        self.tuple_state = isinstance(obs_reset.state, tuple)
        self.episode_history = AttrDict(
            ep_step=torch.zeros(self.config.num_envs), # (N,)
            hidden=(self.rssm.initial(batch_size=self.config.num_envs, dtype=torch.float32, detach_learned=True), torch.zeros(self.config.num_envs, self.env.num_actions, dtype=torch.float32)), 
            state=obs_reset.state,
            episodes=[AttrDict(
                states=tuple([s[env_i]] for s in obs_reset.state) if self.tuple_state else [obs_reset.state[env_i]],
                actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                rewards=[obs_reset.reward[env_i]],
                dones=[obs_reset.done[env_i]],
                is_firsts=[obs_reset.is_first[env_i]]
            ) for env_i in range(self.config.num_envs)]
        )

        # Update Traj Buffer
        for env_i in range(self.config.num_envs):
            sample = []
            if self.tuple_state:
                for s_i, s in enumerate(obs_reset.state):
                    sample.append(s[env_i])
            else:
                sample.append(obs_reset.state[env_i])
            sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
            sample.append(obs_reset.reward[env_i])
            sample.append(obs_reset.done[env_i])
            sample.append(obs_reset.is_first[env_i])
            buffer_infos = self.replay_buffer.append_step(sample, env_i)

        # Add Buffer Infos
        for key, value in buffer_infos.items():
            self.add_info(key, value)

    def on_train_begin(self):

        # Pre Fill Buffer
        if self.config.pre_fill_steps > 0 and self.replay_buffer.num_steps < self.config.pre_fill_steps:
            print("Prefill dataset with {} steps".format(self.config.pre_fill_steps))
            while self.replay_buffer.num_steps < self.config.pre_fill_steps:
                self.env_step()

        # Thread Env Step Loop
        if self.config.thread_env_step:
            self.locks_env_step = [torch.multiprocessing.Lock() for _ in range(3)]
            self.locks_env_step[0].acquire()
            self.locks_env_step[2].acquire()
            threading.Thread(target=self.env_step_loop).start()

    def compile(self):
        
        # Compile World Model
        model_params = itertools.chain(self.repr_net.parameters(), self.rssm.parameters(), self.r_net.parameters(), self.obs_net.parameters(), self.discount_net.parameters())
        self.world_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": model_params, "lr": self.config.model_lr, "grad_max_norm": self.config.model_grad_max_norm, "eps": self.config.model_eps}, 
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Actor Model
        self.actor_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.p_net.parameters(), "lr": self.config.actor_lr, "grad_max_norm": self.config.actor_grad_max_norm, "eps": self.config.actor_eps},
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Value Model
        self.value_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.v_net.parameters(), "lr": self.config.value_lr, "grad_max_norm": self.config.value_grad_max_norm, "eps": self.config.value_eps}, 
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Model Step
        self.model_step = self.world_model.optimizer.param_groups[0]["lr_scheduler"].model_step

        # Optimizer
        self.optimizer = {"world_model": self.world_model.optimizer, "actor_model": self.actor_model.optimizer, "value_model": self.value_model.optimizer}

        # Set Compiled to True
        self.compiled = True

    def env_step_loop(self):

        self.locks_env_step[1].acquire()

        # Env Step
        if self.config.correct_train_ratio:
            num_env_steps = (self.config.batch_size * self.config.L) / (self.config.env_step_period * self.config.num_envs) # correct num_env_steps
        else:
            num_env_steps = (self.config.batch_size * self.config.L) / self.config.env_step_period # old num_env_steps

        while 1:

            # Train Step Check
            # has 1
            self.locks_env_step[0].acquire()
            self.locks_env_step[1].release()
            self.locks_env_step[2].acquire()
            # has 0, 2
            self.locks_env_step[0].release()
            self.locks_env_step[1].acquire()
            self.locks_env_step[2].release()
            # has 1

            if self.config.debug > 0:
                start = time.time()
                print("env_step start")
            
            # Env step every n model step
            if 0 < num_env_steps < 1:
                model_step_period = 1 / num_env_steps
                if self.model_step % model_step_period == 0:
                    with torch.cuda.amp.autocast(enabled=self.config.precision!=torch.float32, dtype=self.config.precision):
                        self.env_step()
            
            # n env steps per model step
            else:
                with torch.cuda.amp.autocast(enabled=self.config.precision!=torch.float32, dtype=self.config.precision):
                    for i in range(int(num_env_steps)):
                        self.env_step()

            if self.config.debug > 0:
                print("env_step end:", time.time()-start, "seconds")

    def env_step(self):

        # set to cuda for compiled model, to fix!
        if self.device != "cuda:0":
            self.to("cuda:0")

        # Eval Mode for BN
        training = self.training
        if self.config.thread_env_step:
            repr_net_env = self.repr_net_env
            rssm_env = self.rssm_env
            p_net_env = self.p_net_env
        else:
            repr_net_env = self.repr_net
            rssm_env = self.rssm
            p_net_env = self.p_net
        repr_net_env.eval()
        rssm_env.eval()
        p_net_env.eval()

        # Env step loop
        for _ in range(self.config.num_env_steps):

            ###############################################################################
            # Forward / Env Step
            ###############################################################################

            # Recover State / hidden
            state = self.episode_history.state
            hidden = self.episode_history.hidden

            # Unpack hidden
            latent, action = hidden

            # Transfer to device
            state = self.transfer_to_device(state)
            latent = self.transfer_to_device(latent)
            action = self.transfer_to_device(action)

            # Forward Policy Network
            with torch.no_grad():

                # Repr State
                emb = repr_net_env(self.preprocess_inputs(state)) # preprocess and augment state
                latent, _ = rssm_env(latent, action, emb, is_first=torch.zeros(self.config.num_envs))
                feat = rssm_env.get_feat(latent)

                # Policy Sample
                action = p_net_env(feat).sample().cpu()

            # Add Noise
            if self.config.noise is not None:
                action += self.noise()

            # Update Hidden
            hidden = (latent, action)

            # Clip Action
            if not self.config.policy_discrete:
                action = action.type(torch.float32).clip(self.env.clip_low, self.env.clip_high)

            # Env Step
            if self.replay_buffer.num_steps < self.config.pre_fill_steps:
                obs = self.env.step(self.env.sample())
            else:
                obs = self.env.step(action.argmax(dim=-1) if self.config.policy_discrete else action)

            ###############################################################################
            # Update Infos / Buffer
            ###############################################################################

            # Update training_infos
            self.action_step += self.env.action_repeat * self.config.num_envs
            self.ep_rewards += obs.reward.to(self.ep_rewards.device)

            # Update History State
            self.episode_history.state = obs.state
            self.episode_history.hidden = hidden
            self.episode_history.ep_step += self.env.action_repeat
            # Update History Episodes
            for env_i in range(self.config.num_envs):
                if not obs.error[env_i]:
                    if self.tuple_state:
                        for s_i, s in enumerate(obs.state):
                            self.episode_history.episodes[env_i].states[s_i].append(s[env_i])
                    else:
                        self.episode_history.episodes[env_i].states.append(obs.state[env_i])
                    self.episode_history.episodes[env_i].actions.append(action[env_i])
                    self.episode_history.episodes[env_i].rewards.append(obs.reward[env_i])
                    self.episode_history.episodes[env_i].dones.append(obs.done[env_i])
                    self.episode_history.episodes[env_i].is_firsts.append(obs.is_first[env_i])

            # Update Traj Buffer
            for env_i in range(self.config.num_envs):
                if not obs.error[env_i]:
                    sample = []
                    if self.tuple_state:
                        for s_i, s in enumerate(obs.state):
                            sample.append(s[env_i])
                    else:
                        sample.append(obs.state[env_i])
                    sample.append(action[env_i])
                    sample.append(obs.reward[env_i])
                    sample.append(obs.done[env_i])
                    sample.append(obs.is_first[env_i])
                    buffer_infos = self.replay_buffer.append_step(sample, env_i)

                    # Add Buffer Infos
                    for key, value in buffer_infos.items():
                        self.add_info(key, value)

            ###############################################################################
            # Reset Env
            ###############################################################################

            # Is_last / Time Limit
            for env_i in range(self.config.num_envs):
                if obs.is_last[env_i]:

                    # Set finished_episode
                    finished_episode = []
                    if self.tuple_state:
                        for s_i in range(len(state)):
                            finished_episode.append(torch.stack(self.episode_history.episodes[env_i].states[s_i], dim=0))
                    else:
                        finished_episode.append(torch.stack(self.episode_history.episodes[env_i].states, dim=0))
                    finished_episode.append(torch.stack(self.episode_history.episodes[env_i].actions, dim=0))
                    finished_episode.append(torch.stack(self.episode_history.episodes[env_i].rewards, dim=0))
                    finished_episode.append(torch.stack(self.episode_history.episodes[env_i].dones, dim=0))
                    finished_episode.append(torch.stack(self.episode_history.episodes[env_i].is_firsts, dim=0))

                    # Copy Episode
                    finished_episode = copy.deepcopy(finished_episode)

                    # Episode State
                    try:
                        for key, value in self.env.envs[env_i].get_episode_state().items():
                            self.add_info(key, value)
                    except Exception as e:
                        pass

                    # Add Infos
                    self.add_info("episode_steps", self.episode_history.ep_step[env_i])
                    self.add_info("episode_reward_total", self.ep_rewards[env_i].item())

                    # Reset Episode Step
                    self.episode_history.ep_step[env_i] = 0

                    # Reset Hidden
                    latent = self.rssm.initial(batch_size=1, dtype=torch.float32, detach_learned=True)
                    action = torch.zeros(self.env.num_actions, dtype=torch.float32)
                    for key in self.episode_history.hidden[0]:
                        self.episode_history.hidden[0][key][env_i] = latent[key].squeeze(dim=0)
                    self.episode_history.hidden[1][env_i] = action

                    # Reset Env
                    obs_reset = self.env.envs[env_i].reset()
                    if self.tuple_state:
                        for s_i, s, in enumerate(obs_reset.state):
                            self.episode_history.state[s_i][env_i] = s
                    else:
                        self.episode_history.state[env_i] = obs_reset.state

                    # Reset Episode History
                    self.episode_history.episodes[env_i] = AttrDict(
                        states=tuple([s] for s in obs_reset.state) if self.tuple_state else [obs_reset.state],
                        actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                        rewards=[obs_reset.reward],
                        dones=[obs_reset.done],
                        is_firsts=[obs_reset.is_first]
                    )

                    # Update Traj Buffer
                    sample = []
                    if self.tuple_state:
                        for s_i, s in enumerate(obs_reset.state):
                            sample.append(s)
                    else:
                        sample.append(obs_reset.state)
                    sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
                    sample.append(obs_reset.reward)
                    sample.append(obs_reset.done)
                    sample.append(obs_reset.is_first)
                    buffer_infos = self.replay_buffer.append_step(sample, env_i)

                    # Add Buffer Infos
                    for key, value in buffer_infos.items():
                        self.add_info(key, value)

                    # Update training_infos
                    self.episodes += 1
                    self.running_rewards.fill_(self.config.running_rewards_momentum * self.ep_rewards[env_i].item() + (1 - self.config.running_rewards_momentum) * self.running_rewards)
                    self.ep_rewards[env_i] = 0.0

        # Default Mode
        repr_net_env.train(mode=training)
        rssm_env.train(mode=training)
        p_net_env.train(mode=training)

    def update_target_networks(self):

        # Update Target Networks
        if 0 <= self.config.critic_ema_decay <= 1:

            # Soft Update
            for param_target, param_net in zip(self.v_target.parameters(), self.v_net.parameters()):
                param_target.mul_(1 - self.config.critic_ema_decay)
                param_target.add_(self.config.critic_ema_decay * param_net.detach())
        else:

            # Hard Update
            if self.model_step % self.config.critic_ema_decay == 0:
                self.v_target.load_state_dict(self.v_net.state_dict())

    def update_env_step_networks(self):

        if self.config.thread_env_step:
            self.repr_net_env.load_state_dict(self.repr_net.state_dict())
            self.rssm_env.load_state_dict(self.rssm.state_dict())
            self.p_net_env.load_state_dict(self.p_net.state_dict())

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Preprocess state (uint8 to float3)
        inputs = self.preprocess_inputs(inputs)

        # Tuple State
        if self.tuple_state:
            inputs = [tuple(i for i in inputs[:-4]), inputs[-4], inputs[-3], inputs[-2], inputs[-1]]

        # Step Time Debug
        if self.config.debug > 0:
            start = time.time()
            print("train_step start")

        ###############################################################################
        # World Train Step
        ###############################################################################

        # World Model Step
        self.set_require_grad(self.p_net, False)
        self.set_require_grad(self.v_net, False)
        self.set_require_grad(self.repr_net, True)
        self.set_require_grad(self.obs_net, True)
        self.set_require_grad(self.rssm, True)
        self.set_require_grad(self.r_net, True)
        self.set_require_grad(self.discount_net, True)
        world_model_batch_losses, world_model_batch_metrics, _ = self.world_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"world_model_" + key: value for key, value in world_model_batch_losses.items()})
        batch_metrics.update({"world_model_" + key: value for key, value in world_model_batch_metrics.items()})
        self.infos.update({"world_model_" + key: value for key, value in self.world_model.infos.items()})

        ###############################################################################
        # Actor Model Step
        ###############################################################################

        self.set_require_grad(self.p_net, True)
        self.set_require_grad(self.v_net, False)
        self.set_require_grad(self.repr_net, False)
        self.set_require_grad(self.obs_net, False)
        self.set_require_grad(self.rssm, False)
        self.set_require_grad(self.r_net, False)
        self.set_require_grad(self.discount_net, False)
        actor_model_batch_losses, actor_model_batch_metrics, _ = self.actor_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"actor_model_" + key: value for key, value in actor_model_batch_losses.items()})
        batch_metrics.update({"actor_model_" + key: value for key, value in actor_model_batch_metrics.items()})
        self.infos.update({"actor_model_" + key: value for key, value in self.actor_model.infos.items()})

        ###############################################################################
        # Value Model Step
        ###############################################################################

        self.set_require_grad(self.p_net, False)
        self.set_require_grad(self.v_net, True)
        self.set_require_grad(self.repr_net, False)
        self.set_require_grad(self.obs_net, False)
        self.set_require_grad(self.rssm, False)
        self.set_require_grad(self.r_net, False)
        self.set_require_grad(self.discount_net, False)
        value_model_batch_losses, value_model_batch_metrics, _ = self.value_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"value_model_" + key: value for key, value in value_model_batch_losses.items()})
        batch_metrics.update({"value_model_" + key: value for key, value in value_model_batch_metrics.items()})
        self.infos.update({"value_model_" + key: value for key, value in self.value_model.infos.items()})

        ###############################################################################
        # Update Target Networks
        ###############################################################################

        # Update value target
        self.update_target_networks()

        # Step Time Debug
        if self.config.debug > 0:
            print("train_step end:", time.time()-start, "seconds")

        ###############################################################################
        # Env Step
        ###############################################################################

        if self.config.thread_env_step:

            # End train Step check
            # has 0, 2
            self.locks_env_step[0].release()
            self.locks_env_step[1].acquire()
            self.locks_env_step[2].release()
            # has 1

            # Update Env Step Networks
            self.update_env_step_networks()
            # print("update_env_step_networks")

            # has 1
            self.locks_env_step[0].acquire()
            self.locks_env_step[1].release()
            self.locks_env_step[2].acquire()
            # has 0, 2

        else:

            # Step Time Debug
            if self.config.debug > 0:
                start = time.time()
                print("env_step start")

            # Env Step
            if self.config.correct_train_ratio:
                num_env_steps = (self.config.batch_size * self.config.L) / (self.config.env_step_period * self.config.num_envs) # correct num_env_steps
            else:
                num_env_steps = (self.config.batch_size * self.config.L) / self.config.env_step_period # old num_env_steps

            # Env step every n model step
            if 0 < num_env_steps < 1:
                model_step_period = 1 / num_env_steps
                if self.model_step % model_step_period == 0:
                    with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                        self.env_step()
            
            # n env steps per model step
            else:
                with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                    for i in range(int(num_env_steps)):
                        self.env_step()

            # Step Time Debug
            if self.config.debug > 0:
                print("env_step end:", time.time()-start, "seconds")

        # Update Infos
        self.infos["episodes"] = self.episodes.item()
        self.infos["running_rewards"] = round(self.running_rewards.item(), 2)
        for env_i in range(self.config.num_envs):
            self.infos["ep_rewards_{}".format(env_i)] = round(self.ep_rewards[env_i].item(), 2)
        self.infos["step"] = self.model_step
        self.infos["action_step"] = self.action_step.item()

        # Built
        if not self.built:
            self.built = True

        return batch_losses, batch_metrics, _
    
    class WorldModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="World Model")
            object.__setattr__(self, "outer", outer)

        def __getattr__(self, name):
            return getattr(self.outer, name)

        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts = inputs
            # states St (B, L, ...) state t returned by env
            # actions At-1 (B, L, A) action t-1 given to env
            # rewards rt (B, L) reward t returned by env
            # dones Dt (B, L) done t returned by env, means agent died
            # is_firsts Ft (B, L) is_first t returned by env, means hidden state / action need reset
            if self.config.debug > 1:
                print("WorldModel:")
                print("states.shape:", tuple(s.shape for s in states) if isinstance(states, tuple) else states.shape)
                print("actions.shape:", actions.shape)
                print("rewards.shape:", rewards.shape) 
                print("dones.shape:", dones.shape)
                print("is_firsts.shape:", is_firsts.shape)

            # Outputs
            outputs = {}

            ###############################################################################
            # Model Forward
            ###############################################################################

            assert actions.shape[1] == self.config.L

            # Forward Representation Network (B, L, D)
            embed = self.repr_net(states)

            # Model Observe (B, L, D)
            posts, priors = self.rssm.observe(embed=embed, prev_actions=actions, is_firsts=is_firsts)

            # Get feat (B, L, Dfeat)
            feats = self.rssm.get_feat(posts)

            # Predict reward (B, L, 1)
            model_rewards = self.r_net(feats)

            # Rec Images (B, L, ...)
            states_pred = self.obs_net(feats)

            # Predict Discounts
            discount_pred = self.discount_net(feats)

            ###############################################################################
            # Model Reconstruction Loss
            ###############################################################################

            # Model Image Loss
            if self.tuple_state:
                for i, s in enumerate(states_pred):
                    self.add_loss("model_rec_{}".format(i), - s.log_prob(states[i].detach()).mean(), weight=self.config.loss_decoder_scale)
            else:
                self.add_loss("model_image", - states_pred.log_prob(states.detach()).mean(), weight=self.config.loss_decoder_scale)

            ###############################################################################
            # Model kl Loss
            ###############################################################################

            # KL
            kl_prior = torch.distributions.kl.kl_divergence(self.rssm.get_dist({k:v.detach() for k, v in posts.items()}), self.rssm.get_dist(priors))
            kl_post = torch.distributions.kl.kl_divergence(self.rssm.get_dist(posts), self.rssm.get_dist({k:v.detach() for k, v in priors.items()}))

            # Add losses, Mean after Free Nats
            self.add_loss("kl_prior", torch.mean(torch.clip(kl_prior, min=self.config.free_nats)), weight=self.config.loss_kl_prior_scale)
            self.add_loss("kl_post", torch.mean(torch.clip(kl_post, min=self.config.free_nats)), weight=self.config.loss_kl_post_scale)

            ###############################################################################
            # Model Reward Loss
            ###############################################################################

            # Model Reward Loss
            self.add_loss("model_reward", - model_rewards.log_prob(rewards.unsqueeze(dim=-1).detach()).mean(), weight=self.config.loss_reward_scale)

            ###############################################################################
            # Model Discount Loss
            ###############################################################################

            # Model Discount Loss
            self.add_loss("model_discount", - discount_pred.log_prob((1.0 - dones).unsqueeze(dim=-1).detach()).mean(), self.config.loss_discount_scale)

            # Flatten and detach post (B, L, D) -> (B*L, D) = (B', D)
            self.outer.detached_posts = {k: v.flatten(start_dim=0, end_dim=1).detach() for k, v in posts.items()}

            return outputs
        
    class ActorModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Actor Model")
            object.__setattr__(self, "outer", outer)

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts  = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Policy Forward
            ###############################################################################

            prev_state = self.detached_posts

            # Model Imagine H next states (B', 1+H, D) with state synchronized actions
            img_states = self.rssm.imagine(p_net=self.p_net, prev_state=prev_state, img_steps=self.config.H)

            # Get feat (B', 1+H, Dfeat)
            feats = self.rssm.get_feat(img_states)

            # Predict rewards (B', 1+H, 1)
            model_rewards = self.r_net(feats)

            # Predict Values (B', 1+H, 1)
            if self.config.target_value_reg:
                values = self.v_net(feats)
            else:
                values = self.v_target(feats)

            # Predict Discounts (B', 1+H, 1)
            discounts = self.discount_net(feats).mode # 0 / 1

            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = (1.0 - dones.flatten(start_dim=0, end_dim=1)).unsqueeze(dim=-1).unsqueeze(dim=-1) # 0 or 1
            discounts = torch.cat([true_first, discounts[:, 1:]], dim=1)

            ###############################################################################
            # Policy Loss
            ###############################################################################

            # (B', 1+H, 1)
            weights = torch.cumprod(self.config.gamma * discounts, dim=1).detach() / self.config.gamma

            # Compute lambda returns (B', H, 1), one action grad lost because of next value
            returns = self.compute_td_lambda(rewards=model_rewards.mode()[:, 1:], values=values.mode()[:, 1:], discounts=self.config.gamma * discounts[:, 1:])
            self.add_info("returns_mean", returns.mean().item())

            # Update Perc
            offset, invscale = self.update_perc(returns)

            # Norm Returns using quantiles ema ~ [0:1]
            normed_returns = (returns - offset) / invscale # 1:H+1
            normed_base = (values.mode()[:, :-1] - offset) / invscale # 0:H

            # advantage (B', H)
            advantage = (normed_returns - normed_base).squeeze(dim=-1)

            # Policy Dist (B', 1+H, A) 
            policy_dist = self.p_net(feats.detach()) 

            # Actor Loss
            if self.config.actor_grad == "dynamics":
                actor_loss = advantage
            elif self.config.actor_grad == "reinforce":
                actor_loss = policy_dist.log_prob(img_states["action"].detach())[:, :-1] * advantage.detach()
            else:
                raise Exception("Unknown actor grad: {}".format(self.actor_grad))
            
            # Add Negative Entropy loss
            policy_ent = policy_dist.entropy()[:, :-1]
            self.add_info("policy_ent", policy_ent.mean().item())
            actor_loss += self.config.eta_entropy * policy_ent

            # Apply weights
            actor_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add loss
            self.add_loss("actor", - actor_loss.mean())  

            self.outer.detached_feats = feats.detach()
            self.outer.detached_returns = returns.detach()
            self.outer.detached_weights = weights.detach()

            return outputs
        
    class ValueModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Value Model")
            object.__setattr__(self, "outer", outer)

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts  = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Value Loss
            ###############################################################################

            feats = self.detached_feats
            returns = self.detached_returns
            weights = self.detached_weights

            # Value (B', H, 1)
            value_dist = self.v_net(feats.detach()[:, :-1])

            # Value Loss
            value_loss = value_dist.log_prob(returns.detach())
            
            # Add Regularization
            if self.config.target_value_reg:
                with torch.no_grad():
                    value_target = self.v_target(feats.detach()[:, :-1]).mode()
                value_loss += self.config.critic_slow_reg_scale * value_dist.log_prob(value_target.detach())

            # Weight loss
            value_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add Loss
            self.add_loss("value", - value_loss.mean())

            return outputs
    
    def update_perc(self, returns):

        # Compute percentiles (,)
        low = torch.quantile(returns.detach(), q=self.config.return_norm_perc_low)
        high = torch.quantile(returns.detach(), q=self.config.return_norm_perc_high)

        # Update percentiles ema
        self.perc_low = self.config.return_norm_decay * self.perc_low + (1 - self.config.return_norm_decay) * low
        self.perc_high = self.config.return_norm_decay * self.perc_high + (1 - self.config.return_norm_decay) * high
        self.add_info("perc_low", self.perc_low.item())
        self.add_info("perc_high", self.perc_high.item())

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def get_perc(self):

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def compute_td_lambda(self, rewards, values, discounts):

        # Asserts
        assert rewards.shape == values.shape, "rewards.shape is {} while values.shape is {}".format(rewards.shape, values.shape)
        assert rewards.shape == discounts.shape, "rewards.shape is {} while discounts.shape is {}".format(rewards.shape, discounts.shape)
        assert rewards.shape[1] == self.config.H, "rewards.shape[1] is {} while self.config.H is {}".format(rewards.shape[1], self.config.H)


        # Init for loop
        interm = rewards + discounts * (1 - self.config.lambda_td) * values
        vals = [values[:, -1]]

        # Recurrence loop
        for t in reversed(range(interm.shape[1])):
            vals.append(interm[:, t] + discounts[:, t] * self.config.lambda_td * vals[-1])

        # Stack and slice init val
        lambda_values = torch.stack(list(reversed(vals))[:-1], dim=1)

        return lambda_values

    def play(self, verbose=False, policy_mode="sample"):

        assert policy_mode in ["sample", "mode", "random"], policy_mode

        # Reset
        obs = self.env_eval.reset()
        state = self.transfer_to_device(obs.state)
        latent = self.transfer_to_device(self.rssm.initial(1, obs.reward.dtype, detach_learned=True))
        action = self.transfer_to_device(torch.zeros(1, self.env.num_actions, dtype=obs.reward.dtype))

        total_rewards = 0
        step = 0

        # Episode loop
        while 1:

            # Sample Action
            with torch.no_grad():
                emb = self.repr_net(self.preprocess_inputs(tuple(s.unsqueeze(dim=0) for s in state) if self.tuple_state else state.unsqueeze(dim=0)))
                latent, _ = self.rssm(latent, action, emb, is_first=torch.zeros(self.config.num_envs))
                feat = self.rssm.get_feat(latent)
                if policy_mode == "sample":
                    action = self.p_net(feat).sample()
                elif policy_mode == "mode":
                    action = self.p_net(feat).mode()
                elif policy_mode == "random":
                    action = self.transfer_to_device(self.env.sample()[:1])

            # Forward Env
            obs = self.env_eval.step(action.argmax(dim=-1).squeeze(dim=0) if self.config.policy_discrete else action.squeeze(dim=0))
            state = self.transfer_to_device(obs.state)
            step += self.env_eval.action_repeat
            total_rewards += obs.reward

            # Done / Time Limit
            if obs.done or step >= self.config.time_limit:
                break

        return total_rewards, step

    def eval_step(self, inputs, targets, verbose=False):

        if self.device != "cuda:0":
            self.to("cuda:0")

        # play
        if self.config.eval_policy_mode == "both":
            outputs = {}
            score, steps = self.play(verbose=verbose, policy_mode="sample")
            outputs["score_sample"] = torch.tensor(score)
            outputs["steps_sample"] = torch.tensor(steps)
            score, steps = self.play(verbose=verbose, policy_mode="mode")
            outputs["score_mode"] = torch.tensor(score)
            outputs["steps_mode"] = torch.tensor(steps)
        else:
            score, steps = self.play(verbose=verbose, policy_mode=self.config.eval_policy_mode)
            outputs = {"score": torch.tensor(score), "steps": torch.tensor(steps)}

        # Update Infos
        for key, value in outputs.items():
            self.infos["ep_{}".format(key)] = value.item()

        return {}, outputs, {}, {}
    
    def log_figure(self, step, inputs, targets, writer, tag, save_image=False): 

        # Eval Mode
        mode = self.training
        self.eval()

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs)

        # Tuple State
        if self.tuple_state:
            inputs = [tuple(i for i in inputs[:-4]), inputs[-4], inputs[-3], inputs[-2], inputs[-1]]

        # Unpack Inputs 
        states, actions, rewards, dones, is_firsts = inputs
        
        # Number of Rows
        if self.tuple_state:
            states = tuple(s[:self.config.log_figure_batch] for s in states)
        else:
            states = states[:self.config.log_figure_batch]
        actions = actions[:self.config.log_figure_batch]
        is_firsts = is_firsts[:self.config.log_figure_batch]

        with torch.no_grad():

            # Forward Representation Network (B, L, D)
            embed = self.repr_net(states)

            ###############################################################################
            # Model
            ###############################################################################

            # Model Observe (B, L, D)
            posts, _ = self.rssm.observe(embed=embed, prev_actions=actions, is_firsts=is_firsts)

            # Get feat (B, L, 2*D)
            feats = self.rssm.get_feat(posts)

            # Rec Images (B, L, ...)
            if self.tuple_state:
                image_pred = self.obs_net(feats)[0].mode()
            else:
                image_pred = self.obs_net(feats).mode()

            ###############################################################################
            # Imaginary
            ###############################################################################

            # Initial State
            if self.config.log_figure_context_frames == 0:
                prev_state = self.transfer_to_device(self.rssm.initial(embed.shape[0], embed.dtype))
            else:
                prev_state = {k: v[:, self.config.log_figure_context_frames-1] for k, v in posts.items()}

            # Model Imagine (B, 1+L-C, D)
            img_states = self.rssm.imagine(p_net=self.p_net, prev_state=prev_state, img_steps=self.config.L-self.config.log_figure_context_frames)

            # Get feat (B', 1+L-C, 2*D)
            feats_img = self.rssm.get_feat(img_states)

            # Concat Context and Img feats
            feats_img = torch.cat([feats[:, :self.config.log_figure_context_frames], feats_img[:, 1:]], dim=1)

            # Rec Images (B, L, ...)
            if self.tuple_state:
                image_img = self.obs_net(feats_img)[0].mode()
            else:
                image_img = self.obs_net(feats_img).mode()

        # Expand is Firsts
        if self.tuple_state:
            is_firsts = is_firsts.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(states[0]) * states[0]
        else:
            is_firsts = is_firsts.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(states) * states

        # Tuple State
        if self.tuple_state:
            states = states[0]

        # Concat Outputs
        outputs = torch.concat([is_firsts, states, image_pred, image_img], dim=1).flatten(start_dim=0, end_dim=1)

        # Add Figure to logs
        if self.rank == 0 and writer != None:
            fig = torchvision.utils.make_grid(outputs, nrow=self.config.L, normalize=True, scale_each=True).cpu()
            writer.add_image(tag, fig, step)

        # Default Mode
        self.train(mode=mode)