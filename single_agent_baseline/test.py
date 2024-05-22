import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import Algorithm, PPOConfig
from ray.rllib.policy.policy import PolicySpec

from utils import plot_2d_hist, plot_heatmaps_rcs_wind, plot_heatmaps_rcs_rainfall, plot_heatmaps_wind_rainfall
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

ray.init()

cdir = '/nas-tmp/Radu/baseline/results/single_agent_baseline/PPO_TrackingEnv_a85c3_00000_0_2024-04-28_12-05-51/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)


env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []

# num_iterations = 100
# for i in range(num_iterations):
#     print(i)
#     obs, _ = env.reset()
#
#     env.wind_speed = 40
#
#     env.altitude = 10
#
#     env.rcs = 1
#
#     env.rainfall_rate = 2.7 * 10e-7
#     done = False
#     while not done:
#         parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1', explore=False)
#
#         actions = {0: parameters_1}
#         # print(f"Parameters: {None} given observation at previous timestep: {obs}")
#         obs, rewards, terminateds, truncateds, _ = env.step(actions)
#
#         done = terminateds["__all__"]
#     pds.append(env.pds)
#     ratios.append(env.ratios)
#     track.append(carpet.firm_track_probability(env.pds))
#
# pds = np.array(pds).reshape(-1)
# pds = np.round(pds, decimals=2)
#
# ratios = np.array(ratios).reshape(-1)
# ratios = np.round(ratios, decimals=2)
#
# track = np.array(track).reshape(-1)
# track = np.round(track, decimals=2)
#
#
# np.savetxt("/project/single_agent_baseline/results/pds.txt", pds)
# np.savetxt("/project/single_agent_baseline/results/ratios.txt", ratios)
# np.savetxt("/project/single_agent_baseline/results/track.txt", track)
# plot_2d_hist(track, ratios)

# env.render_with_variance(pds=pds, ratios=ratios, track_probs=track)
# env.render_hist(pds=pds, ratios=ratios, track_probs=track)
# env.render_hist_treshold(pds=pds, ratios=ratios, track_probs=track)


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

                actions = {0: parameters_1}
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

                actions = {0: parameters_1}
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

                actions = {0: parameters_1}
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
