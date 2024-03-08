import copy
import math
from datetime import datetime, timedelta

import numpy as np
from gymnasium.spaces import Dict, Discrete
from matplotlib import pyplot as plt
from ray.rllib.policy.policy import torch, Policy
from scipy.constants import c
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from config import param_dict
from prototype.prototype.utils import generate_waveform, make_scene, simulate_target_with_scene_profile, doppler_processing, CFAR


def reward(truth, prediction, max_ua_range, max_ua_velocity):
    max_dist = math.dist([-max_ua_range, -max_ua_velocity], [max_ua_range, max_ua_velocity])
    reward = 0
    for t in truth:
        for p in prediction:
            # map to distances to 0 1 range ( a reward of 0 for max distance and 1 for 0 distance) could be non linear scale to not let it gamify
            if len(p) != 0:
                reward += 1 - math.dist(t, p) / max_dist
            else:
                reward = 0
                break
    return reward

    # if False:
    #     agents = ['baseline',
    #               ]
    #     action_space = Dict(
    #         {'pulse_duration': Discrete(5, start=1), 'PRI': Discrete(5, start=1), 'n_pulses': Discrete(30, start=10)})
    #     observation_space = copy.deepcopy(action_space)
    #     # observation_space['measurement'] = Box(low=np.array([-1e3, -1e5]), high=np.array([1e3, 1e5]))
    #     # print(observation_space.sample()['measurement'])
    #     env_config = {
    #         "ts": 10,
    #         'agents': agents,
    #         # Actions -> [pulse_duration, n_pulses, bandwidth, PRF]
    #         'action_space': action_space,
    #         # observe actions of other agents, and previous measurement
    #         'observation_space': observation_space
    #     }
    #
    #     policies = {
    #         "baseline": PolicySpec(),
    #     }
    #
    #
    #     # Policy to agent mapping function
    #     def mapping_fn(agent_id, episode, worker, **kwargs):
    #         return 'baseline'
    #
    #
    #     config = (
    #         PPOConfig().environment(env=MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
    #         .rollouts(num_rollout_workers=6)
    #         .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
    #         .framework("torch")
    #         # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
    #         .resources(num_cpus_per_worker=1, num_gpus=0)
    #         .training(train_batch_size=8, sgd_minibatch_size=4, num_sgd_iter=10)
    #         # .environment(disable_env_checking=True)
    #
    #     )
    #
    #     ppo = config.build()
    #     ppo.restore(
    #         'prototype\\baseline\\results\\PPO_2024-02-21_17-13-36\\PPO_MultiAgentTrackingEnv_2e025_00000_0_num_sgd_iter=10,sgd_minibatch_size=4,train_batch_size=8_2024-02-21_17-13-41\\params.pkl')


action_space = Dict(
    {'pulse_duration': Discrete(5, start=1), 'PRI': Discrete(5, start=1), 'n_pulses': Discrete(30, start=10)})
observation_space = copy.deepcopy(action_space)
my_policy = Policy.from_checkpoint(
    "C:\\Users\\gaghir\\ray_results\\PPO_2024-02-22_14-07-03\\PPO_MultiAgentTrackingEnv_49d10_00000_0_num_sgd_iter=10,sgd_minibatch_size=4,train_batch_size=8_2024-02-22_14-07-10\\params.pkl")

start_time = datetime.now()

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(np.random.uniform(0, 200))])

# 1d model
truth = GroundTruthPath([GroundTruthState([np.random.uniform(0, 500), 1], timestamp=start_time)])

for k in range(1, 11):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
measurements = []
ground_truth = []
obs = observation_space.sample()
for i, state in enumerate(truth):
    ground_truth.append([state.state_vector[0], state.state_vector[1]])

    parameters = my_policy.compute_single_action(obs, policy_id='baseline')
    obs = parameters

    ranges = torch.tensor([state.state_vector[0]])  # torch.Tensor([10E3,50E3,100E3])
    velocities = torch.tensor([state.state_vector[1]])  # torch.Tensor([20,-250,675])
    # amplitudes = (torch.Tensor([1,0.2,0.5]) + 1j*torch.Tensor([0,0.9,0.5]))
    amplitudes = torch.tensor([1.0]) * torch.exp(1j * torch.normal(0, 1, ranges.shape))

    bandwidth = parameters.get('bandwidth', 10e6)
    pulse_duration = param_dict["pulse_duration"][max(1, parameters.get('pulse_duration', 1))]
    n_pulses = max(parameters.get('n_pulses', 30), 1)
    pri = param_dict["PRI"][max(1, parameters.get('PRI', 1))]

    fs = parameters.get('bandwidth', 4 * bandwidth)

    fc = parameters.get('carrier_frequency', 1e9)

    num_samples = int(np.ceil(pulse_duration * fs))

    # wait time in samples
    wait_time = int(np.ceil(pri * fs))

    total_duration = (pulse_duration + wait_time / fs)

    t = torch.linspace(0, pulse_duration, num_samples)

    max_unamb_range = c * (total_duration - pulse_duration) / 2
    max_unamb_vel = c / (4 * fc * total_duration)

    signal = generate_waveform(bandwidth, pulse_duration, n_pulses, t, wait_time)

    # plt.plot(torch.real(signal[0, :]))
    # plt.show()
    nfft_range = 2 * signal.shape[1] - 1
    nfft_doppler = 1024

    scene = make_scene(amplitudes, ranges, velocities, max_unamb_range, max_unamb_vel,
                       nfft_range, nfft_doppler)

    coherent_gain = int(np.ceil(pulse_duration * fs)) * signal.shape[-2]

    coherent_gain_db = 20 * torch.log10(torch.Tensor([coherent_gain]))
    X = simulate_target_with_scene_profile(signal, scene, num_pulses=n_pulses)
    image = doppler_processing(signal, X, nfft_range, nfft_doppler)

    RDout = 20 * torch.log10(1E-16 + torch.abs(image))
    RDout = RDout - coherent_gain_db

    print(f'SNR Loss: {torch.max(RDout)} dB')
    # plt.imshow(RDout[0, :, nfft_range // 2:], aspect='auto',
    #            extent=[0, max_unamb_range * 1E-3, max_unamb_vel, -max_unamb_vel])
    # plt.clim(-60, 0)
    # plt.colorbar()
    # plt.plot()
    # plt.show()

    detections = CFAR(image=image, max_unamb_range=max_unamb_range, max_unamb_vel=max_unamb_vel,
                      nfft_doppler=nfft_doppler, nfft_range=nfft_range, alpha=5, plot=False)
    if len(detections) != 0:
        measurements.append([detections[0][0], detections[0][1]])

        print("Reward: ", reward([ground_truth[i]], [measurements[-1]], max_unamb_range, max_unamb_vel))
print("Found target ", len(measurements), " times")

ground_truth = np.array(ground_truth)
measurements = np.array(measurements)

plt.scatter(ground_truth[1:, 0], ground_truth[1:, 1], label='Truth')
plt.scatter(measurements[1:, 0], measurements[1:, 1], label='Measured')
plt.legend()
plt.xlabel("Range")
plt.ylabel("Velocity")
plt.show()
