import pprint

import numpy as np
import ray
from gymnasium.spaces import Dict, Box, MultiDiscrete
from gymnasium.spaces import Discrete
from ray import tune, air, train
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.policy.policy import PolicySpec
from ray.tune.schedulers import ASHAScheduler

from tracking_env import TrackingEnv

agents = ['baseline']

# pulse duration -> 10 - 50 us
# pRI - prime would be nice, [2,4] kHz
action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'PRI': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'n_pulses': MultiDiscrete(nvec=[21, 21, 21], start=[10, 10, 10])})
observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'PRI': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'n_pulses': MultiDiscrete(nvec=[21, 21, 21], start=[10, 10, 10]),
     'PD': Box(low=-1, high=1),
     'ratio': Box(low=0, high=100)})

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
    PPOConfig().environment(env=TrackingEnv, env_config=env_config, clip_actions=True)
    .rollouts(num_rollout_workers=3)
    # .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_gpus=0, num_cpus_per_worker=2)
    .training(train_batch_size=tune.grid_search([128, 256, 512]), sgd_minibatch_size=tune.grid_search([32, 64]), num_sgd_iter=tune.grid_search([20, 30]))
    .environment(disable_env_checking=True)
    # config = (
    #     PPOConfig().environment(env=TrackingEnv, env_config=env_config, clip_actions=True)
    #     .rollouts(num_rollout_workers=20)
    #     # .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    #     .framework("torch")
    #     # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    #     .resources(num_gpus=1, num_cpus_per_worker=2)
    #     .training(train_batch_size=512, sgd_minibatch_size=128, num_sgd_iter=30)
    #     .environment(disable_env_checking=True)

)

stop = {
    "training_iteration": 100,
    # "time_budget_s":
    # "episode_reward_mean": 10,
    # "episodes_total": 900
}

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='episode_reward_mean',
    mode='max',
    # grace_period=10,
    # reduction_factor=3,
    # brackets=1,
)
# can i write results directly to nas-tmp?
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1,
                             name="carpetsim", checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=20, checkpoint_at_end=True)),
    tune_config=tune.TuneConfig(metric='episode_reward_mean', mode='max', ),
).fit()

best_result = results.get_best_result(metric='episode_reward_mean', mode='min', scope='all')

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})
agent = Algorithm.from_checkpoint(best_result.checkpoint)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=config["env_config"])

obs, _ = env.reset()

truncated = False

while not truncated:
    parameters = agent.compute_single_action(obs)
    # print(f"Parameters: {parameters} given observation at previous timestep: {obs}")
    obs, reward, _, truncated, _ = env.step(parameters)

env.render()
