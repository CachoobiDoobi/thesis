import copy
import pprint
import random

import numpy as np
import ray
from gymnasium.spaces import Dict
from gymnasium.spaces import Discrete, Box
from ray import tune, air
from ray.rllib.algorithms import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from tracking_env import MultiAgentTrackingEnv

agents = ['baseline',
          ]

# pulse duration -> 10 - 50 us
# pRI - prime would be nice, [2,4] kHz
action_space = Dict(
    {'pulse_duration': Discrete(5, start=1), 'PRI': Discrete(5, start=1), 'n_pulses': Discrete(30, start=10)})
observation_space = copy.deepcopy(action_space)
# observation_space['measurement'] = Box(low=np.array([-1e3, -1e5]), high=np.array([1e3, 1e5]))
# print(observation_space.sample()['measurement'])
env_config = {
    "ts": 10,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

policies = {
    "baseline": PolicySpec(),
}


# Policy to agent mapping function
def mapping_fn(agent_id, episode, worker, **kwargs):
    return 'baseline'


config = (
    PPOConfig().environment(env=MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=5)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_cpus_per_worker=1, num_gpus=1)
    .training(train_batch_size=tune.choice([8]), sgd_minibatch_size=tune.choice([4]), num_sgd_iter=tune.choice([10]))
    # .environment(disable_env_checking=True)

)

stop = {
    # "training_iteration": 1,
    # "time_budget_s": 60
}

# algo = config.build()
# result = algo.train()
# print(result)
# is it ok to train like this?
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1),
    tune_config=tune.TuneConfig(metric="episode_reward_mean", mode='max',time_budget_s=1800),
).fit()

best_result = results.get_best_result()

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episodes_total"
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})
