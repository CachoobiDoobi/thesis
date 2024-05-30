import logging
import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from utils import plot_heatmaps_rcs_wind, plot_heatmaps_rcs_rainfall, plot_heatmaps_wind_rainfall
from tracking_env import TrackingEnv

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

ray.init()

cdir = '/nas-tmp/Radu/baseline/results/single_agent_baseline/PPO_TrackingEnv_37af9_00000_0_2024-05-29_09-02-46/checkpoint_000000'
agent = Algorithm.from_checkpoint(cdir)

@ray.remote
def run_simulation(env_config, agent, r, w, rainfall_rate, num_iterations, alt):
    env = TrackingEnv(env_config=env_config)
    pds = 0
    ratios = 0
    track = 0

    for _ in range(num_iterations):
        obs, _ = env.reset()
        env.wind_speed = w
        env.rcs = r
        env.rainfall_rate = rainfall_rate
        env.altitude = alt

        done = False
        while not done:
            parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
            actions = {0: parameters_1}
            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            done = terminateds["__all__"]

        pds += np.mean(env.pds)
        ratios += np.mean(env.ratios)
        track += np.mean(carpet.firm_track_probability(env.pds))

    return pds / num_iterations, ratios / num_iterations, track / num_iterations

def create_tasks(parameter_grid, num_iterations, alt):
    tasks = []
    for (r, w, rainfall_rate) in parameter_grid:
        tasks.append(run_simulation.remote(env_config, agent, r, w, rainfall_rate, num_iterations, alt))
    return tasks

rcs = np.linspace(0.1, 5, num=20)
wind_speed = np.linspace(start=0, stop=18, num=20)
rainfall_rate = np.linspace(start=0, stop=(2.7 * 10e-7) / 25, num=20)
num_iterations = 100

# Create parameter grids
parameter_grid_rcs_wind = [(r, w, rainfall_rate[-1]) for r in rcs for w in wind_speed]
parameter_grid_rcs_rainfall = [(r, wind_speed[-1], r_rate) for r in rcs for r_rate in rainfall_rate]
parameter_grid_wind_rainfall = [(rcs[0], w, r_rate) for w in wind_speed for r_rate in rainfall_rate]

# Run simulations in parallel
tasks_rcs_wind = create_tasks(parameter_grid_rcs_wind, num_iterations, 15)
tasks_rcs_rainfall = create_tasks(parameter_grid_rcs_rainfall, num_iterations, 15)
tasks_wind_rainfall = create_tasks(parameter_grid_wind_rainfall, num_iterations, 15)

results_rcs_wind = ray.get(tasks_rcs_wind)
results_rcs_rainfall = ray.get(tasks_rcs_rainfall)
results_wind_rainfall = ray.get(tasks_wind_rainfall)

# Aggregate results
def aggregate_results(results, shape):
    pds = np.zeros(shape)
    ratios = np.zeros(shape)
    track = np.zeros(shape)
    for idx, (pds_val, ratios_val, track_val) in enumerate(results):
        i = idx // shape[1]
        j = idx % shape[1]
        pds[i, j] = pds_val
        ratios[i, j] = ratios_val
        track[i, j] = track_val
    return pds, ratios, track

pds_rcs_wind, ratios_rcs_wind, track_rcs_wind = aggregate_results(results_rcs_wind, (20, 20))
plot_heatmaps_rcs_wind(pds_rcs_wind, ratios_rcs_wind, track_rcs_wind)

pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall = aggregate_results(results_rcs_rainfall, (20, 20))
plot_heatmaps_rcs_rainfall(pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall)

pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall = aggregate_results(results_wind_rainfall, (20, 20))
plot_heatmaps_wind_rainfall(pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall)
