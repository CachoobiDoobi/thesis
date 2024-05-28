import logging
import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from train import train
from utils import plot_heatmaps_rcs_wind, plot_heatmaps_rcs_rainfall, plot_heatmaps_wind_rainfall
from tracking_env import TrackingEnv

# Initialize Ray
ray.init()

# Define the agent, action space, and observation space
agents = [0]
n_bursts = 6

action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     })
observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     'PD': Box(low=0, high=1),
     'ratio': Box(low=0, high=100),
     'r_hat': Box(low=0, high=1e5),
     'v_hat': Box(low=0, high=1e3),
     'v_wind': Box(low=0, high=40),
     'alt': Box(low=10, high=30)
     }
)

env_config = {
    "ts": 20,
    'agents': agents,
    'action_space': action_space,
    'observation_space': observation_space
}

env = TrackingEnv(env_config=env_config)

# Histogram rendering
num_iterations = 20

@ray.remote
def compute_histogram():
    pd, ratio = ray.get(train.remote(wind_speed=40, rcs=1, rainfall_rate=2.8 * 10e-7))
    track_prob = carpet.firm_track_probability(pd)
    return pd, ratio, track_prob

# Launch tasks in parallel
hist_tasks = [compute_histogram.remote() for _ in range(num_iterations)]
hist_results = ray.get(hist_tasks)

# Unpack the results
pds, ratios, track = zip(*hist_results)

# Render histograms
env.render_hist(pds=pds, ratios=ratios, track_probs=track)
env.render_hist_treshold(pds=pds, ratios=ratios, track_probs=track)


# Function to handle computation
@ray.remote
def compute_metrics(params, num_iterations, fixed_params):
    temp_pds, temp_ratios, temp_track = 0, 0, 0
    futures = [train.remote(**params) for _ in range(num_iterations)]
    results = ray.get(futures)

    for pd, ratio in results:
        temp_pds += np.mean(pd)
        temp_ratios += np.mean(ratio)
        temp_track += np.mean(carpet.firm_track_probability(pd))

    return temp_pds / num_iterations, temp_ratios / num_iterations, temp_track / num_iterations


def run_experiment(param_grid, fixed_params, plot_func, num_iterations):
    results_shape = (len(param_grid[0]), len(param_grid[1]))
    pds = np.zeros(results_shape)
    ratios = np.zeros(results_shape)
    track = np.zeros(results_shape)

    tasks = [
        compute_metrics.remote({**fixed_params, param_names[0]: param1, param_names[1]: param2}, num_iterations,
                               fixed_params)
        for param1 in param_grid[0]
        for param2 in param_grid[1]
    ]

    results = ray.get(tasks)

    index = 0
    for i in range(results_shape[0]):
        for j in range(results_shape[1]):
            pds[i, j], ratios[i, j], track[i, j] = results[index]
            index += 1

    plot_func(pds, ratios, track)


# Experiment 1: RCS vs Wind Speed
param_names = ['rcs', 'wind_speed']
rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(0, 40, num=20)
fixed_params = {'rainfall_rate': 2.8 * 10e-7}
run_experiment((rcs, wind_speed), fixed_params, plot_heatmaps_rcs_wind, num_iterations=5)

# Experiment 2: RCS vs Rainfall Rate
rainfall_rate = np.linspace(0, 2.8 * 1e-6, num=20)
fixed_params = {'wind_speed': 40}
run_experiment((rcs, rainfall_rate), fixed_params, plot_heatmaps_rcs_rainfall, num_iterations=5)

# Experiment 3: Wind Speed vs Rainfall Rate
fixed_params = {'rcs': 1}
run_experiment((wind_speed, rainfall_rate), fixed_params, plot_heatmaps_wind_rainfall, num_iterations=5)

# Shutdown Ray
ray.shutdown()
