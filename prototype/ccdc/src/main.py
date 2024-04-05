import os
import pprint

import gymnasium.spaces.utils
import numpy as np
import ray
from gymnasium.spaces import Dict, Box, MultiDiscrete
from gymnasium.spaces import Discrete
from ray import tune, air, train
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.schedulers import ASHAScheduler

from prototype.ccdc.src.centralized_critic import CentralizedCritic
from prototype.ccdc.src.critic import TorchCentralizedCriticModel
from tracking_env import MultiAgentTrackingEnv

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
     'ratio': Box(low=0, high=100),
     'r_hat': Box(low=0, high=1e5),
     'v_hat': Box(low=0, high=1e3)
     }
)

# action_space = gymnasium.spaces.utils.flatten_space(action_space)
# observation_space = gymnasium.spaces.utils.flatten_space(observation_space)

agents = [0, 1]

env_config = {
    "ts": 10,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

ray.init(local_mode=True)


ModelCatalog.register_custom_model(
    "cc_model",
    TorchCentralizedCriticModel

)

config = (
    PPOConfig()
    .experimental(_enable_new_api_stack=False)
    .environment(MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
    .framework('torch')
    .rollouts(batch_mode="complete_episodes", num_rollout_workers=3)
    .training(model={"custom_model": "cc_model"})
    .multi_agent(
        policies={
            "pol1": (
                None,
                observation_space,
                action_space,
                # `framework` would also be ok here.
                PPOConfig.overrides(framework_str='torch'),
            ),
            "pol2": (
                None,
                observation_space,
                action_space,
                # `framework` would also be ok here.
                PPOConfig.overrides(framework_str='torch'),
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "pol1"
        if agent_id == 0
        else "pol2",
    )
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .training(train_batch_size=128, sgd_minibatch_size=32, num_sgd_iter=20)
)

stop = {
    "training_iteration": 50,
    # "timesteps_total": args.stop_timesteps,
    # "episode_reward_mean": 9,
}

tuner = tune.Tuner(
    CentralizedCritic,
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1),
)
results = tuner.fit()


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

env = MultiAgentTrackingEnv(env_config=config["env_config"])

obs, _ = env.reset()

done = False
while not done:
    #TODO change observation at timestep 0 to something relevant
    parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
    parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

    actions = {0: parameters_1, 1: parameters_2}
    # print(f"Parameters: {parameters} given observation at previous timestep: {obs}")
    obs, rewards, terminateds, truncateds, _ = env.step(actions)

    done = terminateds["__all__"]
env.render()
