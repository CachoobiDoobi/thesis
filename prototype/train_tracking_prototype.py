import random
from gymnasium.spaces import Dict

import ray
from gymnasium.spaces import Discrete, Box
from ray.rllib.algorithms import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import register_env

from prototype.dummy.dummy_env import MultiAgentArena
from prototype.tracking_env import MultiAgentTrackingEnv


def env_creator(env_config):
    return MultiAgentTrackingEnv(env_config)


env_config = {"ts": 10,
              'agents': ['snr'],
              'action_space': Dict({'pulse_duration': Box(low=0, high=1), 'PRF': Box(0, 1), 'n_pulses': Discrete(20)}),
              'observation_space': Dict({})}

check_env(MultiAgentArena())

ray.init()

# Register the environment
register_env("multi", env_creator)

policies = {
    # Use the PolicySpec namedtuple to specify an individual policy:
    "agent1": PolicySpec(), "agent2": PolicySpec()
}

# Policy to agent mapping function
mapping_fn = lambda agent_id, episode, worker, **kwargs: random.choice(env_config['agents'])

config = (
    PPOConfig().environment(env="multi")
    .rollouts(num_rollout_workers=7)
    .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    .framework("torch")
    .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    .resources(num_cpus_per_worker=1, num_gpus=0, num_gpus_per_worker=0)
)
algo = config.build()

for _ in range(5):
    print(algo.train()['episode_reward_mean'])  # 3. train it,

print(algo.evaluate())  # 4. and evaluate it.
