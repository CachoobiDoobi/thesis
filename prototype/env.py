from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentArena(MultiAgentEnv):
    def __init__(self, config=None):
        """ Config takes in width, height, and ts """
        super().__init__()
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 10)
        self.height = config.get("height", 10)
        self._agent_ids = ["agent1", "agent2"]
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 100)

        self.observation_space = MultiDiscrete([self.width * self.height,
                                                self.width * self.height])
        # 0=up, 1=right, 2=down, 3=left.
        self.action_space = Discrete(4)

        # Reset env.
        self.reset()

    def reset(self,  *, seed=None, options=None):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [0, 0]  # upper left corner
        self.agent2_pos = [self.height - 1, self.width - 1]  # lower bottom corner

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # Reset agent1's visited fields.
        self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs(), {}

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """

        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # Agent2 always moves first.
        # events = [collision|agent1_new_field]
        events = self._move(self.agent2_pos, action["agent2"], is_agent1=False)
        events = self._move(self.agent1_pos, action["agent1"], is_agent1=True)

        # Useful for rendering.
        self.collision = "collision" in events

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        # Determine rewards based on the collected events:
        r1 = -1.0 if "collision" in events else 1.0 if "agent1_new_field" in events else -0.5
        r2 = 1.0 if "collision" in events else -0.1

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        terminateds = {
            "agent1": False,
            "agent2": False,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": False,
        }

        truncateds = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }

        return obs, rewards, terminateds, truncateds,  {}  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        ag1_discrete_pos = self.agent1_pos[0] * self.width + \
                           (self.agent1_pos[1] % self.width)
        ag2_discrete_pos = self.agent2_pos[0] * self.width + \
                           (self.agent2_pos[1] % self.width)
        return {
            "agent1": np.array([ag1_discrete_pos, ag2_discrete_pos]),
            "agent2": np.array([ag2_discrete_pos, ag1_discrete_pos]),
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if (is_agent1 and coords == self.agent2_pos) or (not is_agent1 and coords == self.agent1_pos):
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 (agent1 tried to run into agent2)
            # OR Agent2 bumped into agent1 (agent2 tried to run into agent1)
            return {"collision"}

        # No agent blocking -> check walls.
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.width:
            coords[1] = self.width - 1

        # If agent1 -> "new" if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            return {"agent1_new_field"}
        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif (r, c) in self.agent1_visited_fields:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(f"{'!!Collision!!' if self.collision else ''}")
        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()


# env = MultiAgentArena()
#
# obs = env.reset()
# print(obs)
# # Agent1 will move down, Agent2 moves up.
# obs, rewards, dones,trunc, infos = env.step(action={"agent1": 2, "agent2": 0})
# print(obs)
# env.render()
#
# print("Agent1's x/y position={}".format(env.agent1_pos))
# print("Agent2's x/y position={}".format(env.agent2_pos))
# print("Env timesteps={}".format(env.timesteps))
