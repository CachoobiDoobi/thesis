import copy

import numpy as np
from gymnasium.spaces import Dict, Box, Discrete
from matplotlib import pyplot as plt
from ray.rllib.algorithms import PPOConfig
from ray.rllib.policy.policy import torch, PolicySpec
from scipy.constants import c

from prototype.baseline.tracking_env import MultiAgentTrackingEnv
from prototype.utils import generate_waveform, make_scene, simulate_target_with_scene_profile, doppler_processing, CFAR

if True:
    agents = ['baseline',
              ]
    action_space = Dict(
        {'pulse_duration': Box(low=1e-7, high=1e-5), 'PRI': Box(1e-4, 1e-3), 'n_pulses': Discrete(20, start=1)})
    observation_space = copy.deepcopy(action_space)
    # observation_space['measurement'] = Box(low=np.array([-1e3, -1e5]), high=np.array([1e3, 1e5]))
    # print(observation_space.sample()['measurement'])
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
        PPOConfig().environment(env=MultiAgentTrackingEnv, env_config=env_config, clip_actions=True)
        .rollouts(num_rollout_workers=6)
        .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
        .framework("torch")
        # .evaluation(evaluation_num_workers=1, evaluation_interval=5)
        .resources(num_cpus_per_worker=1, num_gpus=0)
        .training(train_batch_size=8, sgd_minibatch_size=4, num_sgd_iter=10)
        # .environment(disable_env_checking=True)

    )

    ppo = config.build()
    ppo.restore(
        "C:\\Users\gaghir\\ray_results\\PPO_2024-02-20_15-10-45\\PPO_MultiAgentTrackingEnv_dba0f_00000_0_num_sgd_iter=10,sgd_minibatch_size=4,train_batch_size=8_2024-02-20_15-10-53\\params.pkl")

    parameters = ppo.compute_single_action(observation_space.sample(), policy_id='baseline')
else:
    parameters = {'PRI': [0.00019], 'n_pulses': 32, 'pulse_duration': [10e-06]}
print(parameters)

ranges = torch.tensor([5])  # torch.Tensor([10E3,50E3,100E3])
velocities = torch.tensor([20])  # torch.Tensor([20,-250,675])
# amplitudes = (torch.Tensor([1,0.2,0.5]) + 1j*torch.Tensor([0,0.9,0.5]))
amplitudes = torch.tensor([1.0]) * torch.exp(1j * torch.normal(0, 1, ranges.shape))

bandwidth = parameters.get('bandwidth', 10e6)
pulse_duration = parameters.get('pulse_duration', [10E-6])[0]
n_pulses = max(parameters.get('n_pulses', 30), 1)
pri = parameters.get('PRI', [5e-4])[0]

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
print(signal.shape)
plt.plot(torch.real(signal[0, :]))
plt.show()
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
plt.imshow(RDout[0, :, nfft_range // 2:], aspect='auto',
           extent=[0, max_unamb_range * 1E-3, max_unamb_vel, -max_unamb_vel])
plt.clim(-60, 0)
plt.colorbar()
plt.plot()
plt.show()

detections = CFAR(image=image, max_unamb_range=max_unamb_range, max_unamb_vel=max_unamb_vel,
                  nfft_doppler=nfft_doppler, nfft_range=nfft_range, alpha=5, plot=True)

print(detections)
