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
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go


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

        self.measurements = []

        self.snrs = []

        self.target_resolution = np.random.choice(np.arange(20, 40))

        self.resolutions = []

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        # reset rewards
        self.episode_reward = 0
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

        self.episode_reward += rewards

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
        prediction = np.array(prediction)
        # single target
        truth = truth[0]
        # map to distances to 0 1 range ( a reward of 0 for max distance and 1 for 0 distance) could be non linear scale to not let it gamify
        if prediction.any():
            reward += 1 - np.linalg.norm(truth - prediction) / max_dist
        else:
            reward = 0
        # gate reward
        if self.episode_reward >= 5:
            # if more than a certain amount of targets found in this episode, the agent can earn extra reward by minimizing the waveform duration
            reward += np.exp(-((doppler_resolution - self.target_resolution) ** 2) / 100)
        return reward

    def interpolate_reward(self, doppler_resolution, reward):
        # the number by which you divide in the exponential changes the width of the curve around the target resolution
        return (1.0 - self.timesteps / self.timestep_limit) * reward + (self.timesteps / self.timestep_limit) * np.exp(
            -((doppler_resolution - self.target_resolution) ** 2) / 100)

    def weighted_reward(self, doppler_resolution, reward, weights=None):
        if weights is None:
            weights = [0.5, 0.5]
        assert np.sum(weights) == 1.0
        return weights[0] * reward + weights[1] * np.exp(
            -((doppler_resolution - self.target_resolution) ** 2) / 100)

    def render(self):
        ground_truth = []
        for state in self.truth:
            ground_truth.append([state.state_vector[0], state.state_vector[1]])

        ground_truth = np.asarray(ground_truth)
        measurements = np.asarray(self.measurements)

        filtered_measurements = remove_outliers(measurements)

        fig, ax = plt.subplots(nrows=2, ncols=2)

        # TODO makes this a line plot and add arrows

        ax[0, 0].plot(ground_truth[:, 0], ground_truth[:, 1], label='Truth', )
        ax[0, 0].scatter(filtered_measurements[:, 0], filtered_measurements[:, 1], label='Measured', marker='x',
                         c=np.arange(len(filtered_measurements)), cmap='viridis')

        for i in range(len(filtered_measurements) - 1):
            start_point = filtered_measurements[i]
            end_point = filtered_measurements[i + 1]

            # Calculate the change in x and y direction
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]

            # Add the arrow
            ax[0, 0].arrow(start_point[0], start_point[1], dx, dy, shape='full', lw=0, length_includes_head=True,
                           head_width=0.2, head_length=0.3, color='red')

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

        ax[1, 1].plot(np.arange(len(self.snrs)), self.snrs)
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("SNR")
        ax[1, 1].set_title("SNR")

        plt.legend()
        plt.savefig('results/results.pdf')
        plt.show()

        # TODO Add an animation

        # Create figure
        fig = go.Figure(
            data=[
                go.Scatter(x=ground_truth[:, 0], y=ground_truth[:, 1], mode='lines+markers', name='True Position'),
                go.Scatter(x=filtered_measurements[:, 0], y=filtered_measurements[:, 1], mode='markers',
                           name='Measurements')],
            layout=go.Layout(
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}])])]
            ),
            frames=[go.Frame(
                data=[go.Scatter(x=ground_truth[:k + 1, 0], y=ground_truth[:k + 1, 1]),
                      go.Scatter(x=filtered_measurements[:k + 1, 0], y=filtered_measurements[:k + 1, 1])]
            ) for k in range(ground_truth.shape[0])]
        )

        # Layout settings
        fig.update_layout(title='True Positions and Measurements Animation', xaxis=dict(range=[0, 10]),
                          yaxis=dict(range=[0, 5]))

        fig.show()


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
