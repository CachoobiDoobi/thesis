import math
import sys
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import torch
from numpy import ndarray
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import time
from simulation import Simulation

class MultiAgentTrackingEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}

        self.timestep_limit = env_config.get("ts", 10)

        self.timesteps = 0

        self.action_space = env_config.get('action_space',
                                           None)

        self.observation_space = env_config.get('observation_space', None)  # self.action_space
        # Should be none or define some default?
        self._agent_ids = env_config.get("agents", None)

        self.rewards = {agent_id: 0 for agent_id in self._agent_ids}

        self.actions = {agent_id: {} for agent_id in self._agent_ids}

        self.truth = None

        self.measurements = []

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        MultiAgentDict, MultiAgentDict]:
        # reset rewards
        self.rewards = dict.fromkeys(self.rewards, 0)
        # reset current timestep
        self.timesteps = 0

        start_time = datetime.now()

        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(100)])

        # 1d model
        truth = GroundTruthPath([GroundTruthState([0, 1], timestamp=start_time)])

        for k in range(1, self.timestep_limit):
            truth.append(GroundTruthState(
                transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k)))

        self.truth = truth
        return self._get_obs(), {}

    def step(self, action_dict):
        self.timesteps += 1

        # keep track of current action
        self.actions = action_dict

        range = torch.tensor([self.truth[self.timesteps - 1].state_vector[0]])
        velocity = torch.tensor([self.truth[self.timesteps - 1].state_vector[1]])

        detections = []
        start_time = time.time()
        for agent in action_dict:
            sim = Simulation(range, velocity, [torch.tensor([1.0 + 0.0j])])
            d = sim.detect(action_dict[agent])
            detections.append(d)
            if len(d) == 0:
                print(agent + " agent has found 0 targets (CFAR/clustering)")
            # TODO what do with measurements
        print("---Time for all agent simulations: %s seconds ---" % (time.time() - start_time))
        # Determine rewards
        rewards = {"baseline": self.reward(list(zip(range.numpy(), velocity.numpy())), detections)}

        self.rewards = dict.fromkeys(self.rewards,
                                     np.array(list(rewards.values())) + np.array(list(self.rewards.values())))

        truncated = self.timesteps >= self.timestep_limit
        terminateds = {agent_id: False for agent_id in self._agent_ids}
        truncateds = {agent_id: truncated for agent_id in self._agent_ids}

        terminateds["__all__"] = False
        truncateds["__all__"] = truncated

        obs = self._get_obs()

        info = {}
        return obs, rewards, terminateds, truncateds, info

    def _get_obs(self):
        obs = self.actions
        # for agent in self._agent_ids:
            # if len(self.measurements) > 0:
            # obs[agent]['measurement'] = self.measurements[-1] if len(self.measurements) > 0 else [0, 0]
        return obs

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if agent_ids is None:
            return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.observation_space.sample() for agent_id in agent_ids}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.observation_space.contains(o) for o in x.values()])

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.action_space.contains(o) for o in x.values()])

    def reward(self, truth, prediction):
        reward = 0
        for t in truth:
            for p in prediction:
                # map to distances to 0 1 range ( a reward of 0 for max distance and 1 for 0 distance) could be non linear scale to not let it gamify
                if len(p) != 0:
                    reward -= math.dist(t, p)
                else:
                    reward = -100
                    break
        return reward
