import ray
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog

from model import TorchCentralizedCriticModel
from src.tracking_env import TrackingEnv

n_bursts = 3

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

cdir = '/nas-tmp/Radu/multi/results/multiagent/CentralizedCritic_MultiAgentTrackingEnv_522d5_00000_0_2024-04-19_17-36-14/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=env_config)

obs, _ = env.reset()

env.wind_speed = 40

env.altitude = 10

env.rcs = 3

env.rainfall_rate = 2.7 * 10e-7

done = False
while not done:
    parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')
    parameters_2 = agent.compute_single_action(obs[1], policy_id='pol2')

    actions = {0: parameters_1, 1: parameters_2}
    print(f"Parameters: {None} given observation at previous timestep: {obs}")
    obs, rewards, terminateds, truncateds, _ = env.step(actions)

    done = terminateds["__all__"]

env.render()
