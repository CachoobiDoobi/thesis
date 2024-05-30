from datetime import datetime, timedelta

import numpy as np
import ray
from carpet import carpet
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms import Algorithm
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from tracking_env import TrackingEnv

agents = [0]
n_bursts = 6

action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     })
observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     'PD': Box(low=0, high=1),
     'ratio': Box(low=0, high=100),
     'r_hat': Box(low=0, high=1e5),
     'v_hat': Box(low=0, high=1e3),
     'v_wind': Box(low=0, high=40),
     'alt': Box(low=10, high=30)
     }
)

env_config = {
    "ts": 20,
    'agents': agents,
    # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    'action_space': action_space,
    # observe actions of other agents, and previous measurement
    'observation_space': observation_space
}

ray.init()

cdir = '/nas-tmp/Radu/baseline/results/single_agent_baseline/PPO_TrackingEnv_37af9_00000_0_2024-05-29_09-02-46/checkpoint_000000'

agent = Algorithm.from_checkpoint(cdir)
# agent.restore(checkpoint_path=os.path.join(checkpoint_dir, "params.pkl"))

start_time = datetime.now()

# transition_model_altitude = CombinedLinearGaussianTransitionModel([ConstantVelocity(100)])

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)])

# truth_alt = GroundTruthPath(
#     [GroundTruthState([np.random.uniform(10, 30), np.random.uniform(1, 3)], timestamp=start_time)])

# 1d model
truth = GroundTruthPath(
    [GroundTruthState([np.random.uniform(1e4, 3e4), np.random.uniform(100, 500)], timestamp=start_time)])

for k in range(1, 20):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
    # truth_alt.append(GroundTruthState(
    #     transition_model_altitude.function(truth_alt[k - 1], noise=True, time_interval=timedelta(seconds=1)),
    #     timestamp=start_time + timedelta(seconds=k)))
    # print(truth[k].state_vector[0], truth[k].state_vector[1])

env = TrackingEnv(env_config=env_config)
pds = []
ratios = []
track = []
waveforms = []
ranges = []
velocities = []
alts = []

obs, _ = env.reset()
env.truth = truth
# env.truth_alt = truth_alt


# max wind speed should be 18m/s
# rainfall rate - 4 mm/hour
# rain clutter set rain range = 0
# diameter = 100
# vary pulses duration between 1 and 40 microseconds
# set filters to be number of pusles everytime
# set 34 max range
# make altitude constant
#
env.wind_speed = 18

env.altitude = 15

env.rcs = 0.5

env.rainfall_rate = (2.8 * 10e-7) / 25
done = False
while not done:
    range = env.truth[env.timesteps].state_vector[0]
    velocity = env.truth[env.timesteps].state_vector[1]
    alt = 15
    ranges.append(range)
    velocities.append(velocity)
    alts.append(alt)

    parameters_1 = agent.compute_single_action(obs[0], policy_id='pol1')

    actions = {0: parameters_1}
    waveforms.append(actions)
    # print(f"Parameters: {None} given observation at previous timestep: {obs}")
    obs, rewards, terminateds, truncateds, _ = env.step(actions)

    done = terminateds["__all__"]

pds.append(env.pds)
ratios.append(env.ratios)
track.append(carpet.firm_track_probability(env.pds))

pds = np.array(pds).reshape(-1)
pds = np.round(pds, decimals=2)

ratios = np.array(ratios).reshape(-1)
ratios = np.round(ratios, decimals=2)

track = np.array(track).reshape(-1)
track = np.round(track, decimals=2)

np.savetxt("/project/single_agent_baseline/results/pds.txt", pds)
np.savetxt("/project/single_agent_baseline/results/ratios.txt", ratios)
np.savetxt("/project/single_agent_baseline/results/track.txt", track)

np.save("/project/single_agent_baseline/results/waveforms.txt", waveforms)


np.savetxt("/project/single_agent_baseline/results/ranges.txt", ranges)
np.savetxt("/project/single_agent_baseline/results/velocities.txt", velocities)
np.savetxt("/project/single_agent_baseline/results/alts.txt", alts)