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
import torchvision

# NeuralNets
from nnet.envs import minerl
from nnet.structs import AttrDict

# Other
import os
import datetime

class MinecraftDiamond:

  """
  
  Minecraft action space (25): 
    noop,
    attack, 
    turn_up, 
    turn_down, 
    turn_left, 
    turn_right, 
    forward, 
    back, 
    left, 
    right, 
    jump, 
    place_dirt, 
    craft_planks, 
    craft_stick, 
    craft_crafting_table, 
    place_crafting_table, 
    craft_wooden_pickaxe, 
    craft_stone_pickaxe, 
    craft_iron_pickaxe, 
    equip_stone_pickaxe, 
    equip_wooden_pickaxe, 
    equip_iron_pickaxe, 
    craft_furnace, 
    place_furnace, 
    smelt_iron_ingot
  
  """

  def __init__(
      self, 
      repeat=1,
      img_size=(64, 64),
      break_speed=100.0,
      gamma=10.0,
      sticky_attack=30,
      sticky_jump=10,
      pitch_limit=(-60, 60), 
      episode_saving_path=None
    ):

    self.state_shape = ([3, img_size[0], img_size[1]], (1178,))

    actions = {
            **BASIC_ACTIONS,
            'craft_planks': dict(craft='planks'),
            'craft_stick': dict(craft='stick'),
            'craft_crafting_table': dict(craft='crafting_table'),
            'place_crafting_table': dict(place='crafting_table'),
            'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
            'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
            'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
            'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
            'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
            'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
            'craft_furnace': dict(nearbyCraft='furnace'),
            'place_furnace': dict(place='furnace'),
            'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
    }

    self.rewards = [
            CollectReward('log', once=1),
            CollectReward('planks', once=1),
            CollectReward('stick', once=1),
            CollectReward('crafting_table', once=1),
            CollectReward('wooden_pickaxe', once=1),
            CollectReward('cobblestone', once=1),
            CollectReward('stone_pickaxe', once=1),
            CollectReward('iron_ore', once=1),
            CollectReward('furnace', once=1),
            CollectReward('iron_ingot', once=1),
            CollectReward('iron_pickaxe', once=1),
            CollectReward('diamond', once=1),
            HealthReward(),
    ]

    self.time_limit = 36000
    self.env = minerl.MinecraftBase(
      actions=actions, 
      repeat=repeat, 
      img_size=img_size, 
      break_speed=break_speed, 
      gamma=gamma, 
      sticky_attack=sticky_attack, 
      sticky_jump=sticky_jump, 
      pitch_limit=pitch_limit
    )
    self.num_actions = len(actions)
    self.action_repeat = self.env._repeat

    # episode_saving_path
    self.episode_saving_path = episode_saving_path
    if self.episode_saving_path is not None:
      if not os.path.isdir(self.episode_saving_path):
        os.makedirs(self.episode_saving_path, exist_ok=True)

    # Env FPS
    self.fps = 20

  def sample(self):

    return torch.randint(0, self.num_actions, size=())

  def process_obs(self, obs):

    # Obs Pixels
    obs_pixels = torch.tensor(obs["image"].copy()).permute(2, 0, 1) # uint8, mem efficient for buffer

    # Reward
    reward = torch.tensor(obs["reward"], dtype=torch.float32)

    # Done
    done = torch.tensor(obs["is_terminal"], dtype=torch.float32)

    # Time Limit
    if self.num_steps >= self.time_limit:
      is_last = torch.tensor(True, dtype=torch.float32)
    else:
      is_last = done

    # Obs lowd
    obs_inventory = torch.tensor(obs["inventory"], dtype=torch.float32)
    obs_inventory_max = torch.tensor(obs["inventory_max"], dtype=torch.float32)
    obs_equipped = torch.tensor(obs["equipped"], dtype=torch.float32)
    obs_health = torch.tensor([obs["health"]], dtype=torch.float32)
    obs_hunger = torch.tensor([obs["hunger"]], dtype=torch.float32)
    obs_breath = torch.tensor([obs["breath"]], dtype=torch.float32)
    obs_reward = torch.tensor([obs["reward"]], dtype=torch.float32)
    obs_lowd = torch.cat([obs_inventory, obs_inventory_max, obs_equipped, obs_health, obs_hunger, obs_breath], dim=0)

    # Obs tuple
    obs = (obs_pixels, obs_lowd)

    return obs, reward, done, is_last
      
  def reset(self):

    # Reset
    obs = self.env._reset()

    # Format Obs
    obs = self.env._obs(obs)

    # Compute Reward
    obs["reward"] = sum([fn(obs, self.env.inventory) for fn in self.rewards])

    # Reset Number Steps
    self.num_steps = 0

    # Process Obs
    state, _, _, _ = self.process_obs(obs)

    # Episode videos
    if self.episode_saving_path is not None:
      self.episode_video = []

    # Reward
    reward = torch.tensor(0.0, dtype=torch.float32)

    # Done
    done = torch.tensor(False, dtype=torch.float32)

    # Is_last
    is_last = torch.tensor(False, dtype=torch.float32)

    # Is First
    is_first = torch.tensor(True, dtype=torch.float32)

    # Episode Score
    self.episode_score = 0.0

    return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

  def step(self, action):

    # Format Action
    action = {"action": action, "reset": False}

    # Env Step
    obs = self.env.step(action)

    # Add to video
    if self.episode_saving_path is not None:
      self.episode_video.append(torch.tensor(obs["image"].copy(), dtype=torch.float32))

    # Compute Reward
    obs["reward"] = sum([fn(obs, self.env.inventory) for fn in self.rewards])

    # Update Num Steps
    self.num_steps += 1

    # Process Obs
    state, reward, done, is_last = self.process_obs(obs)

    # Update Episode Score
    self.episode_score += reward

    # Save Episode
    if done and self.episode_saving_path is not None:

      # Stack videos
      self.episode_video = torch.stack(self.episode_video, dim=0)

      # Datetime
      date_time_score = str(datetime.datetime.now()).replace(" ", "_") + "_" + str(self.episode_score)

      # Save Videos
      torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}.mp4".format(date_time_score)), video_array=self.episode_video, fps=self.fps, video_codec="libx264")

    # Is First
    is_first = torch.tensor(False, dtype=torch.float32)

    return AttrDict(state=state, reward=reward, done=done, is_first=is_first, is_last=is_last)

  def get_episode_state(self):

    items = [
      'log', 
      'planks', 
      'stick', 
      'crafting_table', 
      'wooden_pickaxe', 
      'cobblestone', 
      'stone_pickaxe', 
      'iron_ore', 
      'furnace', 
      'iron_ingot', 
      'iron_pickaxe', 
      'diamond'
    ]

    episode_state = {}
    for item in items:
      episode_state[item] = self.env.inventory[item].item()

    return episode_state

BASIC_ACTIONS = {
    'noop': dict(),
    'attack': dict(attack=1),
    'turn_up': dict(camera=(-15, 0)),
    'turn_down': dict(camera=(15, 0)),
    'turn_left': dict(camera=(0, -15)),
    'turn_right': dict(camera=(0, 15)),
    'forward': dict(forward=1),
    'back': dict(back=1),
    'left': dict(left=1),
    'right': dict(right=1),
    'jump': dict(jump=1, forward=1),
    'place_dirt': dict(place='dirt'),
}

class CollectReward:

  def __init__(self, item, once=0, repeated=0):
    self.item = item
    self.once = once
    self.repeated = repeated
    self.previous = 0
    self.maximum = 0

  def __call__(self, obs, inventory):
    current = inventory[self.item]
    if obs['is_first']:
      self.previous = current
      self.maximum = current
      return 0
    reward = self.repeated * max(0, current - self.previous)
    if self.maximum == 0 and current > 0:
      reward += self.once
    self.previous = current
    self.maximum = max(self.maximum, current)
    return reward
  
class HealthReward:

  def __init__(self, scale=0.01):
    self.scale = scale
    self.previous = None

  def __call__(self, obs, inventory=None):
    health = obs['health']
    if obs['is_first']:
      self.previous = health
      return 0
    reward = self.scale * (health - self.previous)
    self.previous = health
    return reward