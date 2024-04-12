import gymnasium as gym
import minigrid
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from gymnasium import logger, spaces

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, DIR_TO_VEC
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

from minigrid.core.world_object import Goal

class RGBAppendPartial(ObservationWrapper):
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
            highlight=self.unwrapped.highlight, tile_size=self.tile_size,
            agent_pov=True
        )

        return {**obs, "rgb": np.swapaxes(rgb_img, 0, 1)}

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

class TextObsWrapper(ObservationWrapper):
    """
    Wrapper to get a text encoding of a partially observable
    agent view as observation.
    """
    def __init__(self, env, hunter_objects=True):
        """A wrapper that makes the image observation a text of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
            hunter_objects: bool. Label blue Balls as Cows and green Balls as Trees.
        """
        super().__init__(env)

        self.obs_shape  = env.observation_space["image"].shape
        self.hunter_objects = hunter_objects

    def observation(self, obs):

        img = obs['image']
        direction = DIR_TO_VEC[obs['direction']]

        carrying = self.env.unwrapped.carrying
        agent_pos = self.env.unwrapped.agent_pos

        agent_view_size = self.env.unwrapped.agent_view_size

        if carrying is None:
            carrying = 'None'
        else:
            carrying = f'{carrying.type.capitalize()}(color={carrying.color})'

        is_partial_view = True
        txt = np.empty(img.shape[:2], dtype='object')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                if IDX_TO_OBJECT[type] == 'agent':
                    txt[i, j] = f'Agent({i}, {j}, direction=({direction[0]}, {direction[1]}), ' \
                                f'carrying={carrying})'
                    is_partial_view = False  # Agent isn't shown in partial view.
                elif IDX_TO_OBJECT[type] == 'key':
                    txt[i, j] = f'Key({i}, {j}, color={IDX_TO_COLOR[color]})'
                elif IDX_TO_OBJECT[type] == 'door':
                    txt[i, j] = f'Door({i}, {j}, color={IDX_TO_COLOR[color]}, ' \
                                f'state={IDX_TO_STATE[state]})'
                elif IDX_TO_OBJECT[type] == 'empty':
                    txt[i, j] = 'empty'
                elif IDX_TO_OBJECT[type] == 'ball':
                    if self.hunter_objects:
                        if IDX_TO_COLOR[color] == 'blue':
                            txt[i, j] = f'Cow({i}, {j})'
                        elif IDX_TO_COLOR[color] == 'green':
                            txt[i, j] = f'Tree({i}, {j})'
                    else:
                        txt[i, j] = f'Ball({i}, {j}, color={IDX_TO_COLOR[color]})'
                else:
                    txt[i, j] = f'{IDX_TO_OBJECT[type].capitalize()}({i}, {j})'
        if is_partial_view:
            i = agent_view_size//2
            j = txt.shape[1]-1
            txt[i, j] = f'Agent({i}, {j}, carrying={carrying})'

        obs['text'] = txt
        # rows = [' ; '.join(row) for row in txt]
        # text_string = ' ;\n'.join(rows)
        # obs['text'] = text_string

        return obs