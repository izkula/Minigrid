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

from minigrid.envs.hunter import HunterEnv

from gymnasium.envs.registration import register

register(
    id="MiniGrid-Hunter-v0",
    entry_point="minigrid.envs:HunterEnv",
    kwargs={"n_cows": 2, "n_trees": 5, "size": 10, "agent_start_pos": (4, 4),
            "stationary_cows": False},
)

# env = gym.make('MiniGrid-Empty-16x16-v0')
# env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
# env = gym.make('MiniGrid-DoorKey-8x8-v0')
env = gym.make('MiniGrid-Hunter-v0')
obs_orig, _ = env.reset(seed=0)


do_partial = False
if do_partial:
    env_txt = RGBAppendPartial(TextObsWrapper(env))
    obs, _ = env_txt.reset(seed=0)
    plt.imshow(obs['image'][:,:,0]), plt.show()
    plt.imshow(obs['rgb']), plt.show()
    print(obs['text'])

    for action in [2, 0, 3]:
        obs, reward, terminated, truncated, info = env_txt.step(action=action)
        plt.imshow(obs['image'][:,:,0]), plt.show()
        plt.imshow(obs['rgb']), plt.show()
        print(f'Action = {action}')
        print('```\nto\n```')
        print(obs['text'])


do_fully = True
if do_fully:
    env_full_txt = RGBAppend(TextObsWrapper(FullyObsWrapper(env)))
    obs, _ = env_full_txt.reset(seed=2)
    # plt.imshow(obs['image'][:,:,0]), plt.show()
    plt.imshow(obs['rgb']), plt.show()
    print(obs['text'])

    # for action in [2, 0, 3, 2]:
    for i in range(10):
        action = np.random.randint(3)
        obs, reward, terminated, truncated, info = env_full_txt.step(action=action)
        # plt.imshow(obs['image'][:,:,0]), plt.show()
        plt.imshow(obs['rgb']), plt.show()
        print(f'Action = {action}')
        print('```\nto\n```')
        print(obs['text'])


print('done')

