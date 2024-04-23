import os
import platform
import pprint
import sys

import ray
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray import tune, air
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.policy.policy import PolicySpec

if platform.system() == 'Linux':
    file_dir = os.path.dirname("/project/src/common/")
    sys.path.append(file_dir)
    print(sys.path)

from src.common.tracking_env import TrackingEnv

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

# TODO test if we made this much longer
env_config = {
    "ts": 20,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

policies = {
    "pol1": PolicySpec(),
}


# Policy to agent mapping function
def mapping_fn(agent_id, episode, worker, **kwargs):
    return 'pol1'


ray.init()

config = (
    PPOConfig().environment(env=TrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=2)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    .resources(num_gpus=0, num_cpus_per_worker=2)
    .training(train_batch_size=512, sgd_minibatch_size=128, num_sgd_iter=30)
    .environment(disable_env_checking=True)

)

stop = {
    "training_iteration": 1,
    # "time_total_s": 3600 * 14
    # "episode_reward_mean": 10,
    # "episodes_total": 900
}

storage = os.path.abspath("results")

results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1,
                             name="single_agent_baseline", storage_path=storage), ).fit()

best_result = results.get_best_result(metric='episode_reward_mean', mode='max', scope='all')

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})
agent = Algorithm.from_checkpoint(best_result.checkpoint)

env = TrackingEnv(env_config=config["env_config"])

obs, _ = env.reset()

env.wind_speed = 10

env.altitude = 10

env.rcs = 3

env.rainfall_rate = 2.7 * 10e-7

done = False
while not done:
    parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
    actions = {0: parameters_1}
    # print(f"Parameters: {parameters} given observation at previous timestep: {obs}")
    obs, rewards, terminateds, truncateds, _ = env.step(actions)

    done = terminateds["__all__"]
env.render()
