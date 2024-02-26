import math
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from simulation import Simulation


class TrackingEnv(gymnasium.Env):
    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}

        self.timestep_limit = env_config.get("ts", 10)

        self.timesteps = 0

        self.action_space = env_config.get('action_space',
                                           None)

        self.observation_space = env_config.get('observation_space', None)  # self.action_space
        # Should be none or define some default?
        # self._agent_ids = env_config.get("agents", None)

        self.rewards = 0

        self.actions = {}

        self.truth = None

        self.measurements = []

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        # reset rewards
        self.rewards = 0
        # reset current timestep
        self.timesteps = 0

        start_time = datetime.now()

        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(np.random.uniform(0, 200))])

        # 1d model
        truth = GroundTruthPath([GroundTruthState([np.random.uniform(0, 5000), 1], timestamp=start_time)])

        for k in range(1, self.timestep_limit):
            truth.append(GroundTruthState(
                transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k)))

        self.truth = truth
        return self._get_obs(), {}

    def step(self, action_dict):
        self.timesteps += 1

        action_dict["pulse_duration"] = max(1, action_dict.get('pulse_duration', 1))
        action_dict["PRI"] = max(1, action_dict.get('PRI', 1))
        action_dict["n_pulses"] = max(action_dict.get('n_pulses', 30), 10)

        # keep track of current action
        self.actions = action_dict
        print("actions ", self.actions)
        range = torch.tensor([self.truth[self.timesteps - 1].state_vector[0]])
        velocity = torch.tensor([self.truth[self.timesteps - 1].state_vector[1]])

        sim = Simulation(range, velocity, torch.tensor([1.0]) * torch.exp(1j * torch.normal(0, 1, range.shape)))
        detection = sim.detect(action_dict)
        if np.count_nonzero(detection) != 0:
            self.measurements.append(detection)
        else:
            print("no detections", self.timesteps)

        # Determine rewards
        max_ua_range, max_ua_velocity = sim.get_max_unambigous()
        rewards = self.reward(np.dstack((range.numpy(), velocity.numpy()))[0], detection.astype(np.float64),
                              max_ua_range, max_ua_velocity)

        truncated = self.timesteps >= self.timestep_limit
        terminated = False

        obs = self._get_obs()

        info = {}
        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        obs = self.actions
        if len(self.measurements) > 0:
            obs['measurement'] = self.measurements[-1]
            # print(obs, type(self.measurements[-1]))
        else:
            obs = self.observation_space.sample()
        return obs

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.action_space.sample() for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if self.timesteps == 0:
            return {}
        if agent_ids is None:
            return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}
        else:
            return {agent_id: self.observation_space.sample() for agent_id in agent_ids}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        print("is this ever CALLED?")
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.observation_space.contains(o) for o in x.values()])

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.action_space.contains(o) for o in x.values()])

    def reward(self, truth, prediction, max_ua_range, max_ua_velocity):
        max_dist = math.dist([-max_ua_range, -max_ua_velocity], [max_ua_range, max_ua_velocity])
        reward = 0
        # print("truth ", truth)
        # print("prediction ", prediction)
        for t in truth:
            for p in prediction:
                # map to distances to 0 1 range ( a reward of 0 for max distance and 1 for 0 distance) could be non linear scale to not let it gamify
                if len(p) != 0:
                    reward += 1 - np.linalg.norm(t - p) / max_dist
                else:
                    reward = 0
        return reward

    def render(self):
        ground_truth = []
        for state in self.truth:
            ground_truth.append([state.state_vector[0], state.state_vector[1]])

        # print("truth", ground_truth)
        # print("measurements", np.array(self.measurements).shape)
        ground_truth = np.array(ground_truth)
        measurements = np.array(self.measurements)[:, 0, 0, :]

        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label='Truth')
        plt.scatter(measurements[:, 0], measurements[:, 1], label='Measured')
        plt.legend()
        plt.xlabel("Range")
        plt.ylabel("Velocity")
        plt.show()
