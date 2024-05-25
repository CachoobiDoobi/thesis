import logging

import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm

from train import train
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
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []

num_iterations = 20
for i in range(num_iterations):
    print(i, "hist")
    pd, ratio = train(wind_speed=40, rcs=1, rainfall_rate=2.8 * 10e-7)

    pds.append(pd)
    ratios.append(ratio)
    track.append(carpet.firm_track_probability(pd))

# env.render_with_variance(pds=pds, ratios=ratios, track_probs=track)
env.render_hist(pds=pds, ratios=ratios, track_probs=track)
env.render_hist_treshold(pds=pds, ratios=ratios, track_probs=track)

# TODO fix color scale so we can compare among all models and heatmaps
# TODO fix scale so it starts from the actual value of the parameter

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(start=0, stop=40, num=20)
rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)

num_iterations = 5

for i, r in enumerate(rcs):
    for j, w in enumerate(wind_speed):
        for k in range(num_iterations):
            print(i*20 + j, k, "rcs vs wind")
            pd, ratio = train(wind_speed=w, rcs=r, rainfall_rate=2.8 * 10e-7)

            pds[i, j] += np.mean(pd)
            ratios[i, j] += np.mean(ratio)
            track[i, j] += np.mean(carpet.firm_track_probability(pd))

pds = pds / num_iterations
ratios = ratios / num_iterations
track = track / num_iterations

plot_heatmaps_rcs_wind(pds, ratios, track)

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

for i, r in enumerate(rcs):
    for j, w in enumerate(rainfall_rate):
        for k in range(num_iterations):
            print(i*20 + j,k, "rcs vs rainfall")
            pd, ratio = train(wind_speed=40, rcs=r, rainfall_rate=w)

            pds[i, j] += np.mean(pd)
            ratios[i, j] += np.mean(ratio)
            track[i, j] += np.mean(carpet.firm_track_probability(pd))

pds = pds / num_iterations
ratios = ratios / num_iterations
track = track / num_iterations

plot_heatmaps_rcs_rainfall(pds, ratios, track)

########################

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

for i, w in enumerate(wind_speed):
    for j, r in enumerate(rainfall_rate):
        for k in range(num_iterations):
            print(i*20 + j,k, "rain vs wind")

            pd, ratio = train(wind_speed=w, rcs=1, rainfall_rate=r)

            pds[i, j] += np.mean(pd)
            ratios[i, j] += np.mean(ratio)
            track[i, j] += np.mean(carpet.firm_track_probability(pd))

pds = pds / num_iterations
ratios = ratios / num_iterations
track = track / num_iterations

plot_heatmaps_wind_rainfall(pds, ratios, track)
