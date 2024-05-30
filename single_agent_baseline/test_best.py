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


def run_simulation(agent, env_config, p1, p2, param_to_vary, num_iterations, fixed_val):
    env = TrackingEnv(env_config=EnvContext(env_config, 0))
    pds = np.zeros((len(p1), len(p2)))
    ratios = np.zeros((len(p1), len(p2)))
    track = np.zeros((len(p1), len(p2)))

    for i, val1 in enumerate(p1):
        for j, val2 in enumerate(p2):
            for _ in range(num_iterations):
                obs, _ = env.reset()

                if param_to_vary == 'rcs_wind':
                    env.rcs = val1
                    env.wind_speed = val2
                    env.rainfall_rate = fixed_val
                elif param_to_vary == 'rcs_rainfall':
                    env.rcs = val1
                    env.rainfall_rate = val2
                    env.wind_speed = fixed_val
                elif param_to_vary == 'wind_rainfall':
                    env.wind_speed = val1
                    env.rainfall_rate = val2
                    env.rcs = fixed_val

                env.altitude = 15

                done = False
                max_reward = 0
                index = -1
                n = 0

                while not done:
                    parameters_1 = agent.compute_single_action(obs[0])
                    actions = {0: parameters_1}
                    obs, rewards, terminateds, truncateds, _ = env.step(actions)

                    if sum(rewards.values()) > max_reward:
                        max_reward = sum(rewards.values())
                        index = n

                    done = terminateds["__all__"]
                    n += 1

                pds[i, j] = env.pds[index]
                ratios[i, j] = env.ratios[index]
                track[i, j] = carpet.firm_track_probability(env.pds)[index]

    return pds, ratios, track


rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(start=0, stop=18, num=20)
rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)
num_iterations = 1

# Run simulations for RCS vs Wind Speed
p_fixed =(2.7 * 10e-7) / 25
pds_rcs_wind, ratios_rcs_wind, track_rcs_wind = run_simulation(agent, env_config, rcs, wind_speed, 'rcs_wind',
                                                               num_iterations, p_fixed)
plot_heatmaps_rcs_wind(pds_rcs_wind, ratios_rcs_wind, track_rcs_wind)

# Run simulations for RCS vs Rainfall Rate
p_fixed =  18
pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall = run_simulation(agent, env_config, rcs, rainfall_rate,
                                                                           'rcs_rainfall', num_iterations, p_fixed)
plot_heatmaps_rcs_rainfall(pds_rcs_rainfall, ratios_rcs_rainfall, track_rcs_rainfall)

# Run simulations for Wind Speed vs Rainfall Rate
p_fixed = 0.1
pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall = run_simulation(agent, env_config, wind_speed,
                                                                              rainfall_rate, 'wind_rainfall',
                                                                              num_iterations, p_fixed)
plot_heatmaps_wind_rainfall(pds_wind_rainfall, ratios_wind_rainfall, track_wind_rainfall)
