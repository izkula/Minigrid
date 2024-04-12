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

class RGBAppend(ObservationWrapper):
    """
    Wrapper to append fully observable RGB image to observation,
    This can be used to check things are working correctly.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.width * tile_size,
                self.unwrapped.height * tile_size,
                3,
            ),
            dtype="uint8",
        )

        # self.observation_space = spaces.Dict(
        #     {**self.observation_space.spaces, "image": new_image_space}
        # )

    def observation(self, obs):
        rgb_img = self.get_frame(
            highlight=self.unwrapped.highlight, tile_size=self.tile_size
        )

        return {**obs, "rgb": np.swapaxes(rgb_img, 0, 1)}

class TextPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to get a text encoding of a partially observable
    agent view as observation.
    """
    def __init__(self, env, tile_size=8):
        """A wrapper that makes the image observation a text of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        # num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        # new_image_space = spaces.Box(
        #     low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        # )
        # self.observation_space = spaces.Dict(
        #     {**self.observation_space.spaces, "image": new_image_space}
        # )
        self.obs_shape = obs_shape

    # def observation(self, obs):
    #     img = obs["image"]
    #     # out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
    #     out = np.zeros(self.obs_shape, dtype="object")
    #
    #     for i in range(img.shape[0]):
    #         for j in range(img.shape[1]):
    #             type = img[i, j, 0]
    #             color = img[i, j, 1]
    #             state = img[i, j, 2]
    #
    #             out[i, j, type] = 1
    #             out[i, j, len(OBJECT_TO_IDX) + color] = 1
    #             out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1
    #
    #     return {**obs, "image": out}

    def observation(self, obs):

        img = obs['image']
        direction = DIR_TO_VEC[obs['direction']]

        carrying = self.env.carrying

        txt = np.empty(img.shape[:2], dtype='object')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                # txt[i, j] = f'{IDX_TO_OBJECT[type].capitalize()}'
                if IDX_TO_OBJECT[type] == 'key':
                    txt[i, j] = f'Key({i}, {j}, color={IDX_TO_COLOR[color]})'
                elif IDX_TO_OBJECT[type] == 'door':
                    txt[i, j] = f'Door({i}, {j}, color={IDX_TO_COLOR[color]}, ' \
                                f'state={IDX_TO_STATE[state]})'
                elif IDX_TO_OBJECT[type] == 'empty':
                    txt[i, j] = 'empty'
                elif IDX_TO_OBJECT[type] == 'agent':
                    if carrying is None:
                        carrying = 'None'
                    txt[i, j] = f'Agent({i}, {j}, direction=({direction[0]}, {direction[1]}), ' \
                                f'carrying={carrying})'
                else:
                    txt[i, j] = f'{IDX_TO_OBJECT[type].capitalize()}({i}, {j})'

        rows = [' ; '.join(row) for row in txt]
        text_string = ' ;\n'.join(rows)
        print(text_string)
        obs['text'] = text_string
        return obs
    # def observation(self, obs):
    #     objects = np.array(
    #         [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
    #     )
    #     agent_pos = self.env.agent_pos
    #     ncol, nrow = self.width, self.height
    #     grid = np.mgrid[:ncol, :nrow]
    #     _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))
    #
    #     grid = np.concatenate([grid, _objects])
    #     grid = np.transpose(grid, (1, 2, 0))
    #     grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
    #     obs["image"] = grid
    #
    #     return obs



# env = gym.make('MiniGrid-Empty-16x16-v0')
# env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
env = gym.make('MiniGrid-DoorKey-8x8-v0')
obs_orig, _ = env.reset(seed=0)

# env_obs = SymbolicObsWrapper(env)
# obs, _ = env_obs.reset()

env = FullyObsWrapper(env)
obs_full, _ = env.reset(seed=0)
plt.imshow(obs_full['image'][:,:,0])
plt.show()


env_full_txt = RGBAppend(TextPartialObsWrapper(FullyObsWrapper(env)))
obs, _ = env_full_txt.reset(seed=0)
plt.imshow(obs['rgb']), plt.show()

obs, reward, terminated, truncated, info = env_full_txt.step(action=2)
plt.imshow(obs['rgb']), plt.show()

obs, reward, terminated, truncated, info = env_full_txt.step(action=2)
plt.imshow(obs['rgb']), plt.show()


env_full_txt = TextPartialObsWrapper(FullyObsWrapper(env))
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



plt.imshow(np.rot90(obs_orig['image'][:,:,0]))
plt.show()

plt.imshow(np.rot90(obs_orig['image'][:,:,0]).T)
plt.show()

print('done')
# Get the mission description

# print(mission_desc)