import nnet
import torch
import os
import json

# Get env_name
env_name = os.environ["env_name"]
print("DreamerV3 selected env_name: {}".format(env_name))

# Override Config
override_config = os.environ.get("override_config", {})
print("override_config:", override_config)
if isinstance(override_config, str):
    override_config = json.loads(override_config)

# Model
model = nnet.models.rl.dreamer.DreamerV3(env_name=env_name, override_config=override_config)
model.compile()

# Training
precision = torch.float16
grad_init_scale = 32.0
num_workers = 8
epochs = model.config.epochs
epoch_length = model.config.epoch_length
save_episodes = True

# Callback Path
callback_path = "callbacks/DreamerV3/{}".format(env_name)

# Replay Buffer
training_dataset = nnet.datasets.replay_buffers.DreamerV3ReplayBuffer(
    num_workers=num_workers,
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L,
    collate_fn=model.config.collate_fn,
    save_trajectories=save_episodes
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_epidodes)