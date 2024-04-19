import os
import pprint

import ray
import torch
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray import tune, air
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.models import ModelCatalog

from centralized_critic import CentralizedCritic
from critic import TorchCentralizedCriticModel
from tracking_env import MultiAgentTrackingEnv

# os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = 1

# pulse duration -> 10 - 50 us
# pRI - prime would be nice, [2,4] kHz

n_bursts = 3
# TODO add RF frequency
action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts)})
observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
      'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'PD': Box(low=0, high=1),
     'ratio': Box(low=0, high=100),
     'r_hat': Box(low=0, high=1e5),
     'v_hat': Box(low=0, high=1e3),
     'v_wind': Box(low=0, high=40),
     'alt': Box(low=10, high=30)
     })

# action_space = gymnasium.spaces.utils.flatten_space(action_space)
# observation_space = gymnasium.spaces.utils.flatten_space(observation_space)

agents = [0, 1]

env_config = {
    "ts": 20,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

ray.init()

ModelCatalog.register_custom_model(
    "cc_model",
    TorchCentralizedCriticModel

)

config = (
    PPOConfig()
    .experimental(_enable_new_api_stack=False)
    .environment(MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
    .framework('torch')
    .rollouts(batch_mode="complete_episodes", num_rollout_workers=20)
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
    .resources(num_gpus=1, num_cpus_per_worker=2)
    .training(train_batch_size=512, sgd_minibatch_size=128, num_sgd_iter=30)
)

stop = {
    # "training_iteration": 1,
    # "timesteps_total": args.stop_timesteps,
    # "episode_reward_mean": 10,
    "time_total_s": 3600 * 2
}

storage = os.path.abspath("results")

tuner = tune.Tuner(
    CentralizedCritic,
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1,
                             storage_path=storage,
                             name="multiagent"),
)
results = tuner.fit()

best_result = results.get_best_result(metric='episode_reward_mean', mode='max', scope='all')

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

env.wind_speed = 40

env.altitude = 10

env.rcs = 3

done = False
while not done:
    parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
    parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

    actions = {0: parameters_1, 1: parameters_2}
    # print(f"Parameters: {parameters} given observation at previous timestep: {obs}")
    obs, rewards, terminateds, truncateds, _ = env.step(actions)

    done = terminateds["__all__"]
env.render()