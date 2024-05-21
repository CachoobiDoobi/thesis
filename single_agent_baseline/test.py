import logging

import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray._private.dict import merge_dicts
from ray.rllib import RolloutWorker
from ray.rllib.algorithms import Algorithm, PPOConfig
from ray.rllib.policy.policy import PolicySpec

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

ray.init()

cdir = '/nas-tmp/Radu/baseline/results/single_agent_baseline/PPO_TrackingEnv_a85c3_00000_0_2024-04-28_12-05-51/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))
policies = {
    "pol1": PolicySpec(),
}


# Policy to agent mapping function
def mapping_fn(agent_id, episode, worker, **kwargs):
    return 'pol1'

config = (
    PPOConfig().environment(env=TrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=20)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    .resources(num_gpus=1, num_cpus_per_worker=2)
    .training(train_batch_size=512, sgd_minibatch_size=128, num_sgd_iter=30)
    .environment(disable_env_checking=True)

)
# Disable exploration during evaluation
evaluation_config = config.copy()
evaluation_config["explore"] = False

# Merge evaluation config with the original config
merged_config = merge_dicts(agent.config, evaluation_config.to_dict())

# Create a RolloutWorker for evaluation
worker = RolloutWorker(
    env_creator=lambda _: agent.env_creator(merged_config["env_config"]),
    policy=agent.get_policy(),
    config=merged_config,
)

env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []

num_iterations = 100
for _ in range(num_iterations):

    obs, _ = env.reset()

    env.wind_speed = 40

    env.altitude = 10

    env.rcs = 1

    env.rainfall_rate = 2.7 * 10e-7
    done = False
    while not done:
        parameters_1 = worker.policy.compute_single_action(obs[0], policy_id='pol1')

        actions = {0: parameters_1}
        # print(f"Parameters: {None} given observation at previous timestep: {obs}")
        obs, rewards, terminateds, truncateds, _ = env.step(actions)

        done = terminateds["__all__"]
    pds.append(env.pds)
    ratios.append(env.ratios)
    track.append(carpet.firm_track_probability(env.pds))

# env.render_with_variance(pds=pds, ratios=ratios, track_probs=track)
# env.render_hist(pds=pds, ratios=ratios, track_probs=track)
# env.render_hist_treshold(pds=pds, ratios=ratios, track_probs=track)


# pds = np.zeros((20, 20))
# ratios = np.zeros((20, 20))
# track = np.zeros((20, 20))
#
# rcs = np.linspace(1, 20, num=20)
# wind_speed = np.linspace(start=0, stop=40, num=20)
# rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)
#
# num_iterations = 100
#
# for i, r in enumerate(rcs):
#     for j, w in enumerate(wind_speed):
#         for _ in range(num_iterations):
#
#             obs, _ = env.reset()
#
#             env.wind_speed = w
#
#             env.rcs = r
#
#             env.rainfall_rate = 2.7 * 10e-7
#
#             env.altitude = 10
#
#             done = False
#             while not done:
#                 parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
#
#                 actions = {0: parameters_1}
#                 # print(f"Parameters: {None} given observation at previous timestep: {obs}")
#                 obs, rewards, terminateds, truncateds, _ = env.step(actions)
#
#                 done = terminateds["__all__"]
#             pds[i, j] += np.mean(env.pds)
#             ratios[i, j] += np.mean(env.ratios)
#             track[i, j] += np.mean(carpet.firm_track_probability(env.pds))
#
# pds = pds / num_iterations
# ratios = ratios / num_iterations
# track = track / num_iterations
#
# plot_heatmaps_rcs_wind(pds, ratios, track)
#
# pds = np.zeros((20, 20))
# ratios = np.zeros((20, 20))
# track = np.zeros((20, 20))
#
# for i, r in enumerate(rcs):
#     for j, w in enumerate(rainfall_rate):
#         for _ in range(num_iterations):
#
#             obs, _ = env.reset()
#
#             env.wind_speed = 40
#
#             env.rcs = r
#
#             env.rainfall_rate = w
#
#             env.altitude = 10
#
#             done = False
#             while not done:
#                 parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
#
#                 actions = {0: parameters_1}
#                 # print(f"Parameters: {None} given observation at previous timestep: {obs}")
#                 obs, rewards, terminateds, truncateds, _ = env.step(actions)
#
#                 done = terminateds["__all__"]
#             pds[i, j] += np.mean(env.pds)
#             ratios[i, j] += np.mean(env.ratios)
#             track[i, j] += np.mean(carpet.firm_track_probability(env.pds))
#
# pds = pds / num_iterations
# ratios = ratios / num_iterations
# track = track / num_iterations
#
# plot_heatmaps_rcs_rainfall(pds, ratios, track)
#
# ########################
#
# pds = np.zeros((20, 20))
# ratios = np.zeros((20, 20))
# track = np.zeros((20, 20))
#
# for i, w in enumerate(wind_speed):
#     for j, r in enumerate(rainfall_rate):
#         for _ in range(num_iterations):
#
#             obs, _ = env.reset()
#
#             env.wind_speed = w
#
#             env.rcs = 1
#
#             env.rainfall_rate = r
#
#             env.altitude = 10
#
#             done = False
#             while not done:
#                 parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
#
#                 actions = {0: parameters_1}
#                 # print(f"Parameters: {None} given observation at previous timestep: {obs}")
#                 obs, rewards, terminateds, truncateds, _ = env.step(actions)
#
#                 done = terminateds["__all__"]
#             pds[i, j] += np.mean(env.pds)
#             ratios[i, j] += np.mean(env.ratios)
#             track[i, j] += np.mean(carpet.firm_track_probability(env.pds))
#
# pds = pds / num_iterations
# ratios = ratios / num_iterations
# track = track / num_iterations
#
# plot_heatmaps_wind_rainfall(pds, ratios, track)
