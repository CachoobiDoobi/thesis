import copy
import os
import pprint

import numpy as np
from gymnasium.spaces import Dict, Box, Sequence
from gymnasium.spaces import Discrete
from ray import tune, air, train
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.policy.policy import PolicySpec

from tracking_env import TrackingEnv

agents = ['baseline',
          ]

# pulse duration -> 10 - 50 us
# pRI - prime would be nice, [2,4] kHz
action_space = Dict(
    {'pulse_duration': Discrete(5, start=1), 'PRI': Discrete(5, start=1), 'n_pulses': Discrete(21, start=10)})
observation_space = copy.deepcopy(action_space)
observation_space['measurement'] = Box(low=np.array([-1e5, -1e3]), high=np.array([1e5, 1e3]), dtype=np.float64)
env_config = {
    "ts": 5,
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
    PPOConfig().environment(env=TrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=6)
    # .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_cpus_per_worker=1, num_gpus=0)
    .training(train_batch_size=8, sgd_minibatch_size=4, num_sgd_iter=10)
    .environment(disable_env_checking=True)

)

stop = {
    # "training_iteration": 1,
    # "time_budget_s": 60
    "episode_reward_mean": 9.99
}

# algo = config.build()
# result = algo.train()
# print(result)
# is it ok to train like this?
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=train.CheckpointConfig(
        checkpoint_frequency=1, checkpoint_at_end=True)),
    tune_config=tune.TuneConfig(metric="episode_reward_mean", mode='max'),
).fit()

best_result = results.get_best_result()

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episodes_total"
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

agent = Algorithm.from_checkpoint(best_result.checkpoint)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=config["env_config"])

obs = {"PRI": 1, "n_pulses": 10, 'pulse_duration': 1}

truncated = False

while not truncated:

    parameters = agent.compute_single_action(obs, policy_id='baseline')
    print(parameters)
    obs, reward, _, truncated, _ = env.step({"baseline": parameters})

    obs = obs["baseline"]
    truncated = truncated["__all__"]

env.render()
