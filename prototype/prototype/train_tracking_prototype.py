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

from prototype.mp_policy import MessagePassingPolicy
from prototype.prototype_policy import PrototypePolicy
from tracking_env import MultiAgentTrackingEnv

agents = ['proto',
          'mp_agent'
          ]
action_space = Dict(
    {'pulse_duration': Box(low=1e-7, high=1e-5), 'PRI': Box(1e-4, 1e-3), 'n_pulses': Discrete(20, start=1)})
observation_space = copy.deepcopy(action_space)
observation_space['measurement'] = Box(low=np.array([-1e3, -1e5]), high=np.array([1e3, 1e5]))

env_config = {
    "ts": 10,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

policies = {
    "agent": PolicySpec(policy_class=PrototypePolicy),
    "mp": PolicySpec(policy_class=MessagePassingPolicy)
}


# Policy to agent mapping function
def mapping_fn(agent_id, episode, worker, **kwargs):
    return 'agent' if agent_id == 'proto' else 'mp'


config = (
    PPOConfig().environment(env=MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=4)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_cpus_per_worker=1.6, num_gpus=1)
    # .training(train_batch_size=1, sgd_minibatch_size=1, num_sgd_iter=5)
    # .environment(disable_env_checking=True)
)

stop = {
    "training_iteration": 1,
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
    tune_config=tune.TuneConfig(metric="episode_reward_mean", mode='max')
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
