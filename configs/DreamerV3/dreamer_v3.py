import nnet
import os
import json

# Extract params from filename
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
precision = model.config.precision
grad_init_scale = model.config.grad_init_scale
epochs = model.config.epochs
epoch_length = model.config.epoch_length
save_trajectories = True

# Callback Path
if os.environ.get("run_name", False):
    callback_path = "callbacks/DreamerV3/{}/{}".format(os.environ["run_name"], env_name)
else:
    callback_path = "callbacks/DreamerV3/{}".format(env_name)

# Replay Buffer
training_dataset = nnet.datasets.replay_buffers.DreamerV3ReplayBuffer(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L,
    collate_fn=model.config.collate_fn,
    save_trajectories=save_trajectories
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_epidodes)