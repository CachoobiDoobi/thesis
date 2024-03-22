import math
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional
import plotly.graph_objects as go
import plotly.io as pio
import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from scipy.constants import c
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from config import param_dict
from carpet_simulation import CarpetSimulation


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

        self.episode_reward = 0

        self.actions = []

        self.truth = None

        self.pds = []

        self.target_resolution = np.random.choice(np.arange(20, 40))

        self.ratios = []

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        # reset rewards
        self.episode_reward = 0
        # reset current timestep
        self.timesteps = 0

        self.actions = []

        self.pds = []

        self.ratios = []

        start_time = datetime.now()

        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(np.random.uniform(100, 600))])

        # 1d model
        truth = GroundTruthPath([GroundTruthState([np.random.uniform(1e4, 5e4), 1], timestamp=start_time)])

        for k in range(1, self.timestep_limit):
            truth.append(GroundTruthState(
                transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k)))

        self.truth = truth

        return self._get_obs(), {}

    def step(self, action_dict):
        self.timesteps += 1

        action_dict["pulse_duration"] = np.clip(action_dict.get('pulse_duration'), a_min=0, a_max=5)
        action_dict["PRI"] = np.clip(action_dict.get('PRI'), a_min=0, a_max=5)
        action_dict["n_pulses"] = np.clip(action_dict.get('n_pulses'), a_min=10, a_max=31)

        # keep track of current action
        self.actions.append(action_dict)

        range = torch.tensor([self.truth[self.timesteps - 1].state_vector[0]])
        velocity = torch.tensor([self.truth[self.timesteps - 1].state_vector[1]])

        sim = CarpetSimulation(ranges=range, velocities=velocity)
        pds = sim.detect(action_dict)

        self.pds.append(pds)

        rewards = self.reward(pds, action_dict)

        self.episode_reward += rewards

        truncated = self.timesteps >= self.timestep_limit
        terminated = False

        obs = self._get_obs()
        info = {}
        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        obs = self.actions[-1] if len(self.actions) > 0 else self.observation_space.sample()
        obs['PD'] = np.array([self.pds[-1]]) if len(self.pds) > 0 else np.array([0])
        obs['ratio'] = np.array([self.ratios[-1]]) if len(self.ratios) > 0 else np.array([0])
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
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.observation_space.contains(o) for o in x.values()])

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.action_space.contains(o) for o in x.values()])

    def reward(self, pds, action_dict):
        reward = np.mean(pds)
        pris = [param_dict['PRI'][pri] for pri in action_dict["PRI"]]
        n_pulses = action_dict['n_pulses']
        durations = pris * n_pulses
        duration = np.sum(durations)
        min_duration = 1 / (2 * 1e9 * self.target_resolution / c)
        ratio = duration / min_duration
        # print(n_pulses, pris, duration, min_duration, ratio)

        self.ratios.append(ratio)
        # gate reward
        if self.episode_reward >= 6.9:
            # if more than a certain amount of targets found in this episode, the agent can earn extra reward by minimizing the waveform duration
            sigma = 0.25  # Adjust the width of the Gaussian curve
            reward += math.exp(-(ratio - 1) ** 2 / (2 * sigma ** 2))  # Gaussian function
        return reward

    def render(self):
        # Create Plotly figure for the first plot (Probability of detection)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(self.timestep_limit), y=self.pds, mode='lines', name='Probability of detection'))
        fig1.update_layout(
            title="Probability of Detection",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        # Save the first plot to a file
        pio.write_image(fig1, 'results/probability_of_detection.pdf')

        # Create Plotly figure for the second plot (Waveform duration ratio)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=np.arange(self.timestep_limit), y=self.ratios, mode='lines', name='Waveform duration ratio'))
        fig2.update_layout(
            title="Waveform Duration Ratio",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        # Save the second plot to a file
        pio.write_image(fig2, 'results/waveform_duration_ratio.pdf')

