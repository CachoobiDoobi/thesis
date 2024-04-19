import math
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from scipy.constants import c
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from carpet_simulation import CarpetSimulation
from config import param_dict


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
        self.agent_ids = list(env_config.get("agents", None))
        self._agent_ids = env_config.get("agents", None)

        self.episode_reward = 0

        self.actions = []

        self.truth = None

        self.pds = []

        self.target_resolution = None

        self.ratios = []

        self.range_uncertainty = None

        self.velocity_uncertainty = None

        self.sim = None

        self.altitude = None

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
        truth = GroundTruthPath([GroundTruthState([np.random.uniform(1e4, 5e4), 100], timestamp=start_time)])

        for k in range(1, self.timestep_limit):
            truth.append(GroundTruthState(
                transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k)))

        self.truth = truth

        self.altitude = np.random.uniform(low=10, high=30)

        self.target_resolution = np.random.choice(np.arange(20, 40))

        self.sim = CarpetSimulation()

        self.wind_speed = np.random.uniform(0, 40)

        self.rcs = np.random.uniform(1, 10)

        self.rainfall_rate = np.random.uniform(0, 100)

        return self._get_obs(), {}

    def step(self, action_dict):
        self.timesteps += 1
        for agent in action_dict:
            parameters = action_dict[agent]
            parameters["pulse_duration"] = np.clip(parameters.get('pulse_duration'), a_min=0, a_max=5)
            parameters["PRI"] = np.clip(parameters.get('PRI'), a_min=0, a_max=5)
            parameters["n_pulses"] = np.clip(parameters.get('n_pulses'), a_min=10, a_max=31)

        # keep track of current action
        self.actions.append(action_dict)

        range = self.truth[self.timesteps - 1].state_vector[0]
        velocity = self.truth[self.timesteps - 1].state_vector[1]

        pds, scnr = self.sim.detect(action_dict=action_dict, range_=range, velocity=velocity, altitude=self.altitude, wind_speed=self.wind_speed, rcs=self.rcs, rainfall_rate=self.rainfall_rate)

        if scnr != 0:
            #print(f"The speed of light {c}, and the scnr{scnr}")
            self.range_uncertainty = c / (2 * 1e6 * np.sqrt(2 * scnr))

            duration = 0
            for agent in self._agent_ids:
                pris = [param_dict['PRI'][pri] for pri in action_dict[agent]["PRI"]]
                n_pulses = action_dict[agent]['n_pulses']
                durations = pris * n_pulses
                duration += np.sum(durations)
            wavelength = c / 3300000000
            self.velocity_uncertainty = wavelength / (2 * duration * np.sqrt(2 * scnr))
        else:
            self.range_uncertainty = 1
            self.velocity_uncertainty = 1

        #print(f"Uncertainty {self.range_uncertainty, self.velocity_uncertainty}")
        self.pds.append(pds)

        rewards = self.reward(pds, action_dict)
        self.episode_reward += sum(rewards.values())

        terminateds = {"__all__": self.timesteps >= self.timestep_limit}
        truncateds = {"__all__": False}

        obs = self._get_obs()
        info = {}
        return obs, rewards, terminateds, truncateds, info

    def _get_obs(self):
        range = self.truth[max(0, self.timesteps - 1)].state_vector[0]
        vel = self.truth[max(0, self.timesteps - 1)].state_vector[1]
        r_hat = np.float32(np.random.normal(range, 50, size=(1,)) if self.timesteps == 0 else abs(np.random.normal(range,
                                                                                                               abs(range * self.range_uncertainty),
                                                                                                               size=(
                                                                                                                   1,))))
        v_hat = np.float32(np.random.normal(vel, 5, size=(1,)) if self.timesteps == 0 else abs(np.random.normal(vel,
                                                                                                            abs(vel * self.velocity_uncertainty),
                                                                                                            size=(1,))))

        alt_hat = self.altitude * np.random.normal(1, 0.25)
        alt_hat = np.array([alt_hat], dtype=np.float32)
        # print(range, vel, r_hat, v_hat, self.range_uncertainty, self.velocity_uncertainty)
        # for now 2 agents
        one = self.agent_ids[0]
        two = self.agent_ids[1]
        obs = dict()

        obs[one] = self.actions[-1][two] if len(self.actions) > 0 else self.observation_space.sample()
        obs[one]['r_hat'] = np.clip(r_hat, a_min=0, a_max=1e5)
        obs[one]['v_hat'] = np.clip(v_hat, a_min=0, a_max=1e3)
        obs[one]['v_wind'] = np.array([self.wind_speed], dtype=np.float32)
        obs[one]['alt'] = np.clip(alt_hat, a_min=10, a_max=30)
        obs[one]['PD'] = np.array([self.pds[-1]], dtype=np.float32) if len(self.pds) > 0 else np.array([0],
                                                                                                       dtype=np.float32)
        obs[one]['ratio'] = np.array([self.ratios[-1]], dtype=np.float32) if len(self.ratios) > 0 else np.array([0],
                                                                                                                dtype=np.float32)

        obs[two] = self.actions[-1][one] if len(self.actions) > 0 else self.observation_space.sample()
        obs[two]['r_hat'] = np.clip(r_hat, a_min=0, a_max=1e5)
        obs[two]['v_hat'] = np.clip(v_hat, a_min=0, a_max=1e3)
        obs[two]['v_wind'] = np.array([self.wind_speed], dtype=np.float32)
        obs[two]['alt'] = np.clip(alt_hat, a_min=10, a_max=30)
        obs[two]['PD'] = np.array([self.pds[-1]], dtype=np.float32) if len(self.pds) > 0 else np.array([0],
                                                                                                       dtype=np.float32)
        obs[two]['ratio'] = np.array([self.ratios[-1]], dtype=np.float32) if len(self.ratios) > 0 else np.array([0],
                                                                                                                dtype=np.float32)
        return obs

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            return {agent_id: dict(self.action_space.sample()) for agent_id in self._agent_ids}
        else:
            return {agent_id: dict(self.action_space.sample()) for agent_id in agent_ids}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        if agent_ids is None:
            return {agent_id: dict(self.observation_space.sample()) for agent_id in self._agent_ids}
        else:
            return {agent_id: dict(self.observation_space.sample()) for agent_id in agent_ids}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.observation_space.contains(o) for o in x.values()])

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if self.timesteps == 0:
            return True
        return reduce(lambda n, m: n and m, [self.action_space.contains(o) for o in x.values()])

    def reward(self, pds, action_dict):
        reward_pd = np.mean(pds)

        duration = 0
        for agent in self._agent_ids:
            pris = [param_dict['PRI'][pri] for pri in action_dict[agent]["PRI"]]
            n_pulses = action_dict[agent]['n_pulses']
            durations = pris * n_pulses
            duration += np.sum(durations)
        min_duration = 1 / (2 * 1e9 * self.target_resolution / c)
        ratio = duration / min_duration

        self.ratios.append(ratio)

        sigma = 0.25
        reward_time = math.exp(-(ratio - 1) ** 2 / (2 * sigma ** 2))  # Gaussian function

        return {0: reward_pd, 1: reward_time}

    def render(self):
        # TODO also make a per agent plot
        # Create Plotly figure for the first plot (Probability of detection)
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=np.arange(self.timestep_limit), y=self.pds, mode='lines', name='Probability of detection'))
        fig1.update_layout(
            title="Probability of Detection",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        # Save the first plot to a file
        pio.write_image(fig1, 'results/probability_of_detection.pdf')

        # Create Plotly figure for the second plot (Waveform duration ratio)
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=np.arange(self.timestep_limit), y=self.ratios, mode='lines', name='Waveform duration ratio'))
        fig2.update_layout(
            title="Waveform Duration Ratio",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        # Save the second plot to a file
        pio.write_image(fig2, 'results/waveform_duration_ratio.pdf')
