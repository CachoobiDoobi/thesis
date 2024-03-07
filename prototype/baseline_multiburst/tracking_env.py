import math
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

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

        self.actions = []

        self.truth = None

        self.measurements = []

        self.snrs = []

        self.target_resolution = np.random.choice(np.arange(20, 40))

        self.resolutions = []

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        # reset rewards
        self.rewards = 0
        # reset current timestep
        self.timesteps = 0

        self.actions = []

        self.resolutions = []

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

        # print("actions ", self.actions)
        range = torch.tensor([self.truth[self.timesteps - 1].state_vector[0]])
        velocity = torch.tensor([self.truth[self.timesteps - 1].state_vector[1]])

        sim = Simulation(range, velocity, torch.normal(0, 1, range.shape) + 1j * torch.normal(0, 1, range.shape))
        detection = sim.detect(action_dict)

        self.snrs.append(sim.snr)
        doppler_resolution = sim.doppler_resolution
        self.resolutions.append(doppler_resolution)

        # check if empty or outside range (ambiguity detection does that some time)
        if np.count_nonzero(detection) != 0 and not (detection[0] > 1e5 or abs(detection[1]) > 1e3):
            self.measurements.append(detection)

        # Determine rewards
        max_ua_range, max_ua_velocity = sim.get_max_unambigous()
        rewards = self.reward(np.dstack((range.numpy(), velocity.numpy()))[0], detection,
                              max_ua_range, max_ua_velocity, doppler_resolution)
        truncated = self.timesteps >= self.timestep_limit
        terminated = False

        obs = self._get_obs()
        info = {}
        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        if self.timesteps == 0:
            obs = self.observation_space.sample()
            obs['measurement'] = np.array([self.truth[0].state_vector[0], self.truth[0].state_vector[1]],
                                          dtype=np.float64)
            return obs
        obs = self.actions[-1]
        obs['target_res'] = self.target_resolution
        obs['SNR'] = np.array([self.snrs[-1]], dtype=np.float32)
        if len(self.measurements) > 0:
            obs['measurement'] = np.asarray(self.measurements[-1], dtype=np.float64)
        else:
            obs['measurement'] = np.array([0, 0], dtype=np.float64)
        # print("obs ", obs)
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

    def reward(self, truth, prediction, max_ua_range, max_ua_velocity, doppler_resolution):
        max_dist = math.dist([-max_ua_range, -max_ua_velocity], [max_ua_range, max_ua_velocity])
        reward = 0
        # print("truth ", truth)
        # print("prediction ", prediction)
        for t in truth:
            for p in prediction:
                # map to distances to 0 1 range ( a reward of 0 for max distance and 1 for 0 distance) could be non linear scale to not let it gamify
                if p.size != 0:
                    reward += 1 - np.linalg.norm(t - p) / max_dist
                else:
                    reward = 0
        # the nuber by which you divide in the exponential changes the width of the curve around the target resolution
        reward = (1 - self.timesteps / self.timestep_limit) * reward + (self.timesteps / self.timestep_limit) * np.clip(
            np.exp(-((doppler_resolution - self.target_resolution) ** 2) / 100), a_min=0, a_max=1)
        return reward

    def render(self):
        ground_truth = []
        for state in self.truth:
            ground_truth.append([state.state_vector[0], state.state_vector[1]])

        ground_truth = np.asarray(ground_truth)
        measurements = np.asarray(self.measurements)

        filtered_measurements = remove_outliers(measurements)

        fig, ax = plt.subplots(nrows=2, ncols=2)

        ax[0, 0].scatter(ground_truth[:, 0], ground_truth[:, 1], label='Truth', c=np.arange(len(ground_truth)),
                         cmap='viridis')
        ax[0, 0].scatter(filtered_measurements[:, 0], filtered_measurements[:, 1], label='Measured', marker='x',
                         c=np.arange(len(filtered_measurements)), cmap='viridis')
        ax[0, 0].legend()
        ax[0, 0].set_xlabel("Range")
        ax[0, 0].set_ylabel("Velocity")
        ax[0, 0].set_title("Tracking Simulation")
        # plt.show()

        pris = np.array([param_dict['PRI'][pri] for action in self.actions for pri in action["PRI"]]).reshape(-1, 3)
        n_pulses = np.array([action['n_pulses'] for action in self.actions]).reshape(-1, 3)
        durations = pris * n_pulses
        durations = np.sum(durations, axis=1)
        min_duration = 1 / (2 * 1e9 * self.target_resolution / c)

        ax[0, 1].plot(np.arange(0, ground_truth.shape[0]), durations / min_duration)
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Waveform duration ratio")
        ax[0, 1].set_title("Waveform duration")
        # plt.show()

        closest_pairs = find_closest_pairs(ground_truth, measurements)
        errors = [(np.linalg.norm(pair[0] - pair[1]) / np.linalg.norm(pair[0])) * 100 for pair in closest_pairs]

        ax[1, 0].plot(errors)
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Estimation error %")
        ax[1, 0].set_title("Error")
        # plt.show()

        ax[1, 1].plot(self.snrs)
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("SNR")
        ax[1, 1].set_title("SNR")

        fig.tight_layout()
        plt.show()


def find_closest_pairs(array1, array2):
    # Ensure inputs are numpy arrays
    a1 = np.array(array1)
    a2 = np.array(array2)

    # Calculate the squared Euclidean distances between all pairs
    # This involves expanding the square and summing over the last dimensions
    distances = np.sum((a1[:, np.newaxis, :] - a2[np.newaxis, :, :]) ** 2, axis=2)

    # Find the index of the closest element in array2 for each element in array1
    min_indices = np.argmin(distances, axis=1)

    # Construct the list of closest pairs
    closest_pairs = [(a1[i], a2[min_indices[i]]) for i in range(len(a1))]

    return closest_pairs


def remove_outliers(data, k=1.5):
    """
    Remove outliers from a 2D numpy array based on the IQR method.

    Parameters:
    - data: 2D numpy array where each row is a point [x, y].
    - k: Multiplier for the IQR. Defaults to 1.5.

    Returns:
    - A 2D numpy array with outliers removed.
    """
    # Initialize a mask for all data points, starting with all set to True
    mask = np.ones(data.shape[0], dtype=bool)

    for i in range(data.shape[1]):  # Iterate over columns (dimensions)
        # Calculate Q1 and Q3
        Q1 = np.percentile(data[:, i], 25)
        Q3 = np.percentile(data[:, i], 75)
        # Calculate IQR
        IQR = Q3 - Q1
        # Update the mask to exclude outliers
        mask &= (data[:, i] >= Q1 - k * IQR) & (data[:, i] <= Q3 + k * IQR)

    return data[mask]
