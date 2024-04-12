from __future__ import annotations

from operator import add
from gymnasium.spaces import Discrete
import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class HunterEnv(MiniGridEnv):
    def __init__(
        self,
        size=12,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_cows=2,
        n_trees=2,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_cows <= size / 2 + 1:
            self.n_cows = int(n_cows)
        else:
            self.n_cows = int(size / 2)

        if n_trees <= size / 2 + 1:
            self.n_trees = int(n_trees)
        else:
            self.n_trees = int(size / 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.cows = []
        for i_cow in range(self.n_cows):
            self.cows.append(Ball())
            self.place_obj(self.cows[i_cow], max_tries=100)

        self.trees = []
        for i_tree in range(self.n_trees):
            self.trees.append(Ball(color="green"))
            self.place_obj(self.trees[i_tree], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        # Update obstacle positions
        for i_cow in range(len(self.cows)):
            old_pos = self.cows[i_cow].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            ### Randomly move up/down/left/right or stay
            # try:
            if True:
                pos_options = np.array([
                                       tuple(map(add, old_pos, (0, 0))),
                                       tuple(map(add, old_pos, (1, 0))),
                                       tuple(map(add, old_pos, (-1, 0))),
                                       tuple(map(add, old_pos, (0, 1))),
                                       tuple(map(add, old_pos, (0, -1)))
                                      ])
                option_probs = np.ones(len(pos_options))/len(pos_options)
                max_tries = 100
                num_tries = 0
                while True:
                    if num_tries > max_tries:
                        raise RecursionError("rejection sampling failed in place_obj")
                    num_tries += 1

                    ind = np.random.choice(np.arange(len(pos_options)), p=option_probs)
                    pos = pos_options[ind]

                    if pos[0]==old_pos[0] and pos[1]==old_pos[1]:
                        break

                    # Don't place the object on top of another object
                    if self.grid.get(*pos) is not None:
                        continue

                    # Don't place the object where the agent is
                    if np.array_equal(pos, self.agent_pos):
                        continue

                    break

                self.put_obj(self.cows[i_cow], pos[0], pos[1])

                # self.place_obj(
                #     self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                # )
                if not (pos[0]==old_pos[0] and pos[1]==old_pos[1]):
                    self.grid.set(old_pos[0], old_pos[1], None)
            # except Exception:
            #     pass

            ### Randomly place in the 3x3 square around current position.
            # try:
            #     self.place_obj(
            #         self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
            #     )
            #     self.grid.set(old_pos[0], old_pos[1], None)
            # except Exception:
            #     pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

def main():
    env = HunterEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()