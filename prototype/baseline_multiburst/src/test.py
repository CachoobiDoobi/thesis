import numpy as np
import ray
from gymnasium.spaces import MultiDiscrete, Dict, Box, Discrete
from ray.rllib.algorithms import Algorithm

from prototype.baseline_multiburst.src.tracking_env import TrackingEnv

action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'PRI': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'n_pulses': MultiDiscrete(nvec=[21, 21, 21], start=[10, 10, 10])})
observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'PRI': MultiDiscrete(nvec=[5, 5, 5], start=[0, 0, 0]),
     'n_pulses': MultiDiscrete(nvec=[21, 21, 21], start=[10, 10, 10]),
     'measurement': Box(low=np.array([0, -1e3]), high=np.array([1e5, 1e3]), dtype=np.float64),
     'target_res': Discrete(40, start=10), 'SNR': Box(-200, 200, dtype=np.float32)})

env_config = {
    "ts": 10,
    'agents': ['baseline'],
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}
ray.init(num_cpus=4, num_gpus=0)

agent = Algorithm.from_checkpoint(r"results/2024-03-18_02-23-44/checkpoint_000000/")
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

env = TrackingEnv(env_config=env_config)

obs, _ = env.reset()

truncated = False

# TODO save plots

while not truncated:
    parameters = agent.compute_single_action(obs)
    print(f"Parameters: {parameters} given observation at previous timestep: {obs}")
    obs, reward, _, truncated, _ = env.step(parameters)

env.render()
