from functools import reduce
from typing import Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict


class MultiAgentTrackingEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}

        self.timestep_limit = env_config.get("ts", 10)
        self.timesteps = 0

        # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
        self.action_space = env_config.get('action_space', None) #Dict({'pulse_duration': Box(low=0, high=1), 'PRF': Box(0, 1), 'n_pulses': Discrete(20)})
        # observe all agent actions? or estimated velocity and position? time  budget per burst
        self.observation_space = env_config.get('observation_space', None)  #self.action_space
        # Should be none or define some default?
        self._agent_ids = env_config.get("agents", None)

        self.rewards = {agent_id: 0 for agent_id in self._agent_ids}

        self.actions = {}

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        # reset rewards
        self.rewards = dict.fromkeys(self.rewards, 0)
        # reset current timestep
        self.timesteps = 0

        return self._get_obs(), {}

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.timesteps += 1

        # An episode is "done" when we reach the time step limit.
        truncated = self.timesteps >= self.timestep_limit

        obs = self._get_obs()

        # Determine rewards
        rewards = {}
        self.rewards = dict.fromkeys(self.rewards, np.array(rewards.values()) + np.array(self.rewards.values()))

        terminated = {agent_id: False for agent_id in self._agent_ids}
        truncated = {agent_id: truncated for agent_id in self._agent_ids}

        terminated["__all__"] = False
        truncated["__all__"] = truncated

        info = {}

        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        return {}

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if agent_ids is None:
            return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.observation_space.sample() for agent_id in agent_ids}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        return reduce(lambda n, m: n and m, [self.observation_space.contains(o) for o in x.values()])

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        return reduce(lambda n, m: n and m, [self.action_space.contains(o) for o in x.values()])