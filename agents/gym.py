import gymnasium as gym
import minigrid
from minigrid.wrappers import SymbolicObsWrapper, ViewSizeWrapper, RGBImgObsWrapper, FullyObsWrapper
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from gymnasium import logger, spaces

import matplotlib.pyplot as plt
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.constants import STATE_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR, DIR_TO_VEC
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

from minigrid.core.world_object import Goal

from agents.custom_wrappers import TextObsWrapper, RGBAppend, RGBAppendPartial


# env = gym.make('MiniGrid-Empty-16x16-v0')
# env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
env = gym.make('MiniGrid-DoorKey-8x8-v0')
obs_orig, _ = env.reset(seed=0)

# env = FullyObsWrapper(env)
# obs_full, _ = env.reset(seed=0)
# plt.imshow(obs_full['image'][:,:,0])
# plt.show()

env_txt = RGBAppendPartial(TextObsWrapper(env))
obs, _ = env_txt.reset(seed=0)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print(obs['text'])

obs, reward, terminated, truncated, info = env_txt.step(action=1)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print('Action = 2')
print(obs['text'])

obs, reward, terminated, truncated, info = env_txt.step(action=2)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print('Action = 2')
print(obs['text'])


env_full_txt = RGBAppend(TextObsWrapper(FullyObsWrapper(env)))
obs, _ = env_full_txt.reset(seed=0)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print(obs['text'])

obs, reward, terminated, truncated, info = env_full_txt.step(action=2)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print('Action = 2')
print(obs['text'])

obs, reward, terminated, truncated, info = env_full_txt.step(action=0)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print('Action = 2')
print(obs['text'])

obs, reward, terminated, truncated, info = env_full_txt.step(action=3)
plt.imshow(obs['image'][:,:,0]), plt.show()
plt.imshow(obs['rgb']), plt.show()
print('Action = 2')
print(obs['text'])





env_full_txt = TextObsWrapper(FullyObsWrapper(env))
obs, _ = env_full_txt.reset(seed=0)

print('```\nto\n```')
obs, reward, terminated, truncated, info = env_full_txt.step(action=0)

env_rgb = RGBImgObsWrapper(env)
rgb, _ = env_rgb.reset(seed=0)
plt.imshow(np.swapaxes(rgb['image'], 0, 1))
plt.show()

rgb, reward, terminated, truncated, info = env_rgb.step(action=0)
plt.imshow(np.swapaxes(rgb['image'], 0, 1))
plt.show()

env_txt = TextPartialObsWrapper(env)
obs, _ = env_txt.reset(seed=0)

# env_obs = SymbolicObsWrapper(env)
# obs, _ = env_obs.reset()

plt.imshow(np.rot90(obs_orig['image'][:,:,0]))
plt.show()

plt.imshow(np.rot90(obs_orig['image'][:,:,0]).T)
plt.show()

print('done')
# Get the mission description

# print(mission_desc)