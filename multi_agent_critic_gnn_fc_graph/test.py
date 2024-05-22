import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog

from model import TorchCentralizedCriticModel
from tracking_env import TrackingEnv
from utils import plot_heatmaps_rcs_rainfall, plot_heatmaps_wind_rainfall, plot_heatmaps_rcs_wind, plot_2d_hist

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

cdir = '/nas-tmp/Radu/cc_gnn_fc_graph/results/cc_gnn_fc_graph/CentralizedCritic_TrackingEnv_b7792_00000_0_2024-05-01_08-13-39/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(start=0, stop=40, num=20)
rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)

num_iterations = 1

for i, r in enumerate(rcs):
    for j, w in enumerate(wind_speed):
        for _ in range(num_iterations):
            print(i * 20 + j)
            obs, _ = env.reset()

            env.wind_speed = w

            env.rcs = r

            env.rainfall_rate = 2.7 * 10e-7

            env.altitude = 10

            done = False
            max_reward = 0
            index = -1
            n = 0
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}
                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)
                if sum(rewards.values()) > max_reward:
                    max_reward = sum(rewards.values())
                    index = n
                done = terminateds["__all__"]
                n += 1
            pds[i, j] = env.pds[index]
            ratios[i, j] = env.ratios[index]
            track[i, j] = carpet.firm_track_probability(env.pds)[index]


plot_heatmaps_rcs_wind(pds, ratios, track)

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

for i, r in enumerate(rcs):
    for j, w in enumerate(rainfall_rate):
        for _ in range(num_iterations):
            print(j * 20 + i)
            obs, _ = env.reset()

            env.wind_speed = 40

            env.rcs = r

            env.rainfall_rate = w

            env.altitude = 10

            done = False
            max_reward = 0
            index = -1
            n = 0
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}
                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)
                if sum(rewards.values()) > max_reward:
                    max_reward = sum(rewards.values())
                    index = n
                done = terminateds["__all__"]
                n += 1
            pds[i, j] = env.pds[index]
            ratios[i, j] = env.ratios[index]
            track[i, j] = carpet.firm_track_probability(env.pds)[index]


plot_heatmaps_rcs_rainfall(pds, ratios, track)

########################

pds = np.zeros((20, 20))
ratios = np.zeros((20, 20))
track = np.zeros((20, 20))

for i, w in enumerate(wind_speed):
    for j, r in enumerate(rainfall_rate):
        for _ in range(num_iterations):
            print(j * 20 + i)
            obs, _ = env.reset()

            env.wind_speed = w

            env.rcs = 1

            env.rainfall_rate = r

            env.altitude = 10

            done = False
            max_reward = 0
            index = -1
            n = 0
            while not done:
                parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
                parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

                actions = {0: parameters_1, 1: parameters_2}
                # print(f"Parameters: {None} given observation at previous timestep: {obs}")
                obs, rewards, terminateds, truncateds, _ = env.step(actions)
                if sum(rewards.values()) > max_reward:
                    max_reward = sum(rewards.values())
                    index = n
                done = terminateds["__all__"]
                n += 1
            pds[i, j] = env.pds[index]
            ratios[i, j] = env.ratios[index]
            track[i, j] = carpet.firm_track_probability(env.pds)[index]

plot_heatmaps_wind_rainfall(pds, ratios, track)
