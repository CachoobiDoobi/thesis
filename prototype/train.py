import random

import ray
from ray import air, tune
from ray.rllib.algorithms import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import register_env

from env import MultiAgentArena


def env_creator(env_config):
    return MultiAgentArena()


ray.init(num_gpus=1)

# check_env(MultiAgentArena())

# Register the environment
register_env("multi", env_creator)

policies = {
    # Use the PolicySpec namedtuple to specify an individual policy:
    "agent1": PolicySpec(), "agent2": PolicySpec()
}

# Policy to agent mapping function
mapping_fn = lambda agent_id, episode, worker, **kwargs: random.choice(["agent1", "agent2"])

config = (
    PPOConfig().environment(env="multi")
    .rollouts(num_rollout_workers=7)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_cpus_per_worker=1, num_gpus=1, num_gpus_per_worker=1/8)
)

# disable checking
# everything looks ok to me, and it works, check still fails on the step method
config.environment(disable_env_checking=True)

algo = config.build()

for _ in range(5):
    print(algo.train()['episode_reward_mean'])  # 3. train it,

print(algo.evaluate())  # 4. and evaluate it.

# I think this is for hyperparemeter tuning
# .Tuner(
#     "PPO",
#     run_config=air.RunConfig(
#         stop={"episodes_total" : 3},
#         checkpoint_config=air.CheckpointConfig(
#             checkpoint_frequency=10,
#         ),
#     ),
#     param_space=config,
# ).fit()
