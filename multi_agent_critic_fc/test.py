import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog

from model import TorchCentralizedCriticModel
from tracking_env import TrackingEnv
from utils import plot_heatmaps_rcs_rainfall, plot_heatmaps_wind_rainfall, plot_heatmaps_rcs_wind

n_bursts = 3

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
agents = [0, 1]

env_config = {
    "ts": 20,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

ray.init()

ModelCatalog.register_custom_model(
    "cc_model",
    TorchCentralizedCriticModel

)

cdir = '/nas-tmp/Radu/cc_fc/results/multiagent_cc_fc/CentralizedCritic_TrackingEnv_a50e2_00000_0_2024-04-29_08-51-18/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []

num_iterations = 1000
for _ in range(num_iterations):

    obs, _ = env.reset()

    env.wind_speed = 40

    env.altitude = 10

    env.rcs = 1

    env.rainfall_rate = 2.7 * 10e-7
    done = False
    while not done:
        parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
        parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

        actions = {0: parameters_1, 1: parameters_2}
        # print(f"Parameters: {None} given observation at previous timestep: {obs}")
        obs, rewards, terminateds, truncateds, _ = env.step(actions)

        done = terminateds["__all__"]
    pds.append(env.pds)
    ratios.append(env.ratios)
    track.append(carpet.firm_track_probability(env.pds))

env.render_hist(pds=pds, ratios=ratios, track_probs=track)
env.render_hist_treshold(pds=pds, ratios=ratios, track_probs=track)

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(start=0, stop=40, num=20)
rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)

num_iterations = 100

for i, r in enumerate(rcs):
    for j, w in enumerate(wind_speed):
        for _ in range(num_iterations):

            obs, _ = env.reset()

            env.wind_speed = w

            env.rcs = r

            env.rainfall_rate = 2.7 * 10e-7

            env.altitude = 10

            done = False
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}
                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)

                done = terminateds["__all__"]
            pds[i, j] += np.mean(env.pds)
            ratios[i, j] += np.mean(env.ratios)
            track[i, j] += np.mean(carpet.firm_track_probability(env.pds))

pds = pds / num_iterations
ratios = ratios / num_iterations
track = track / num_iterations

plot_heatmaps_rcs_wind(pds, ratios, track)

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

for i, r in enumerate(rcs):
    for j, w in enumerate(rainfall_rate):
        for _ in range(num_iterations):

            obs, _ = env.reset()

            env.wind_speed = 40

            env.rcs = r

            env.rainfall_rate = w

            env.altitude = 10

            done = False
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}
                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)

                done = terminateds["__all__"]
            pds[i, j] += np.mean(env.pds)
            ratios[i, j] += np.mean(env.ratios)
            track[i, j] += np.mean(carpet.firm_track_probability(env.pds))

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
        for _ in range(num_iterations):

            obs, _ = env.reset()

            env.wind_speed = w

            env.rcs = 1

            env.rainfall_rate = r

            env.altitude = 10

            done = False
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}

                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)

                done = terminateds["__all__"]
            pds[i, j] += np.mean(env.pds)
            ratios[i, j] += np.mean(env.ratios)
            track[i, j] += np.mean(carpet.firm_track_probability(env.pds))

pds = pds / num_iterations
ratios = ratios / num_iterations
track = track / num_iterations

plot_heatmaps_wind_rainfall(pds, ratios, track)
