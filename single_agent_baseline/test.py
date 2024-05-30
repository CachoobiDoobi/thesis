import logging
import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.env_context import EnvContext
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

checkpoint_dir = '/nas-tmp/Radu/baseline/results/single_agent_baseline/PPO_TrackingEnv_37af9_00000_0_2024-05-29_09-02-46/checkpoint_000000'

# Initialize the PPO algorithm with the environment configuration
config = {
    "env": TrackingEnv,
    "env_config": env_config,
}

# Create and restore the PPO agent
agent = PPO(env=TrackingEnv, config=config)
agent.restore(checkpoint_dir)

def run_simulation(agent, env_config, p1, p2, p_fixed, num_iterations, alt):
    env = TrackingEnv(env_config=EnvContext(env_config, 0))
    pds = np.zeros((len(p1), len(p2)))
    ratios = np.zeros((len(p1), len(p2)))
    track = np.zeros((len(p1), len(p2)))

    for i, r in enumerate(p1):
        for j, w in enumerate(p2):
            for _ in range(num_iterations):
                obs, _ = env.reset()

                env.wind_speed = w
                env.rcs = r
                env.rainfall_rate = p_fixed
                env.altitude = alt

                done = False
                while not done:
                    parameters_1 = agent.compute_single_action(obs[0])
                    actions = {0: parameters_1}
                    obs, rewards, terminateds, truncateds, _ = env.step(actions)
                    done = terminateds["__all__"]

                pds[i, j] += np.mean(env.pds)
                ratios[i, j] += np.mean(env.ratios)
                track[i, j] += np.mean(carpet.firm_track_probability(env.pds))

    return pds / num_iterations, ratios / num_iterations, track / num_iterations

rcs = np.linspace(0.1, 5, num=20)
wind_speed = np.linspace(start=0, stop=18, num=20)
rainfall_rate = np.linspace(start=0, stop=(2.7 * 10e-7) / 25, num=20)
num_iterations = 100

# Run simulations sequentially
pds_rcs_wind, ratios_rcs_wind, track_rcs_wind = run_simulation(agent, env_config, rcs, wind_speed, rainfall_rate[-1], num_iterations, 15)
plot_heatmaps_rcs_wind(pds_rcs_wind, ratios_rcs_wind, track_rcs_wind)

pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall = run_simulation(agent, env_config, rcs, rainfall_rate, wind_speed[-1], num_iterations, 15)
plot_heatmaps_rcs_rainfall(pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall)

pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall = run_simulation(agent, env_config, wind_speed, rainfall_rate, rcs[0], num_iterations, 15)
plot_heatmaps_wind_rainfall(pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall)
