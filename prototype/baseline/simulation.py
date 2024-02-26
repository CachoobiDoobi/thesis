import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import pi
from pyapril.caCfar import CA_CFAR
from scipy.constants import c
from torch import Tensor
from sklearn.cluster import DBSCAN
from pyapril import caCfar
import time
from config import param_dict


class Simulation:
    def __init__(self, ranges: [Tensor], velocities: [Tensor], amplitudes: [Tensor]):
        self.ranges = ranges
        self.velocities = velocities
        self.amplitudes = amplitudes
        self.max_unamb_range = 0
        self.max_unamb_velocity = 0

    def detect(self, parameters):

        # start_time = time.time()
        # for now single burst
        bandwidth = parameters.get('bandwidth', 10e6)
        pulse_duration = param_dict["pulse_duration"][max(1, parameters.get('pulse_duration', 1))]
        n_pulses = max(parameters.get('n_pulses', 30), 1)
        pri = param_dict["PRI"][max(1, parameters.get('PRI', 1))]

        fs = parameters.get('bandwidth', 4 * bandwidth)

        fc = parameters.get('carrier_frequency', 1e9)

        self.num_samples = int(np.ceil(pulse_duration * fs))

        # wait time in samples
        wait_time = int(np.ceil(pri * fs))

        total_duration = (pulse_duration + wait_time / fs)

        t = torch.linspace(0, pulse_duration, self.num_samples)

        max_unamb_range = c * (total_duration - pulse_duration) / 2
        max_unamb_vel = c / (4 * fc * total_duration)

        self.max_unamb_range = max_unamb_range
        self.max_unamb_velocity = max_unamb_vel

        signal = self.generate_waveform(bandwidth, pulse_duration, n_pulses, t, wait_time)

        nfft_range = 2 * signal.shape[1] - 1
        nfft_doppler = 1024

        scene = self.make_scene(self.amplitudes, self.ranges, self.velocities, max_unamb_range, max_unamb_vel,
                                nfft_range, nfft_doppler)

        coherent_gain = int(np.ceil(pulse_duration * fs)) * signal.shape[-2]

        # coherent_gain_db = 20 * torch.log10(torch.Tensor([coherent_gain]))

        X = self.simulate_target_with_scene_profile(signal, scene, num_pulses=n_pulses)
        image = self.doppler_processing(signal, X, nfft_range, nfft_doppler)

        detections = self.CFAR(image=image, max_unamb_range=max_unamb_range, max_unamb_vel=max_unamb_vel,
                               nfft_doppler=nfft_doppler, nfft_range=nfft_range)
        # detections = self.cfar_april(image, max_unamb_range, max_unamb_vel, nfft_doppler, nfft_range)
        # print("--- %s seconds ---" % (time.time() - start_time))
        return detections[0] if len(detections) > 0 else []

    # def cfar_april(self, x, max_unamb_range, max_unamb_vel, nfft_doppler, nfft_range):
    #     r_res = max_unamb_range / nfft_range
    #     v_res = max_unamb_vel / nfft_doppler
    #     k_width = int(np.ceil(r_res * 8))
    #     k_height = int(np.ceil(v_res * 18))
    #     inner_width = k_width // 2
    #     inner_height = k_height // 2
    #
    #     x = x.squeeze(0)
    #     print("starting detection")
    #     cfar = CA_CFAR([k_height, k_width, inner_height, inner_width], 22, x.shape)
    #     detection = cfar(x)
    #     print(detection)
    #     return detection

    def CFAR(self, image, max_unamb_range, max_unamb_vel, nfft_range, nfft_doppler, alpha=5, plot=False):
        image = image.reshape(1, 1, nfft_doppler, nfft_range)
        noise = torch.normal(0, 1000, image.shape)
        # SINR = SNR(image, noise)

        # print(f'SINR: {SINR}')
        image = torch.abs(image + noise)
        # create kernel
        r_res = max_unamb_range / nfft_range
        v_res = max_unamb_vel / nfft_doppler
        k_width = 40  # int(np.ceil(r_res * 8))
        k_height = 1  # int(np.ceil(v_res * 5))
        kernel = torch.ones((1, 1, k_height, k_width))
        inner_width = 6  # k_width // 4
        inner_height = 0  # k_height // 4
        kernel[0, 0, :, (k_width - inner_width) // 2: (k_width + inner_width) // 2] = 0
        kernel = kernel / kernel.sum()
        # convolve image
        # print("IMage size", image.shape)
        convd = torch.nn.functional.conv2d(input=image, weight=kernel, padding='valid', stride=1, )
        # compare with estimated noise power
        threshold = image[0, 0, :convd.shape[2], :convd.shape[3]] > convd * alpha

        # some reshaping
        final_x = threshold.reshape(nfft_doppler - k_height + 1, nfft_range - k_width + 1).detach().numpy()
        final_x = np.where(final_x == True)
        final_x = np.dstack((final_x[1] * max_unamb_range * 2 / nfft_range - max_unamb_range,
                             final_x[0] * max_unamb_vel * 2 / nfft_doppler - max_unamb_vel))[0]

        # clustering
        return self.clustering(image=image, final_x=final_x, max_unamb_range=max_unamb_range,
                               max_unamb_vel=max_unamb_vel, nfft_doppler=nfft_doppler, nfft_range=nfft_range,
                               plot=False)

    def clustering(self, image, final_x, max_unamb_range, max_unamb_vel, nfft_doppler, nfft_range, plot):
        if len(final_x) == 0:
            print("NO TARGET FOUND(CFAR)")
            return []
        db = DBSCAN(eps=50, min_samples=10).fit(final_x)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        if plot:
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k

                xy = final_x[class_member_mask & core_samples_mask]

                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

                xy = final_x[class_member_mask & ~core_samples_mask]
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                )

            plt.title(f"Estimated number of clusters: {n_clusters_}")
            plt.show()
        xs = []
        ys = []
        for k in unique_labels:
            class_member_mask = labels == k

            xy = final_x[class_member_mask & core_samples_mask]

            r = ((xy[:, 0] + max_unamb_range) * nfft_range / (2 * max_unamb_range)).astype(int)
            v = ((xy[:, 1] + max_unamb_vel) * nfft_doppler / (2 * max_unamb_vel)).astype(int)
            powers = image[0, 0, v, r]
            x = (powers * xy[:, 0]).sum() / powers.sum()
            y = (powers * xy[:, 1]).sum() / powers.sum()
            if not (x.isnan() or y.isnan()):
                xs.append(x.item())
                ys.append(y.item())
        if plot:
            plt.scatter(xs, ys)
            plt.show()
        return np.dstack((xs, ys))

    def generate_waveform(self, bandwidth, pulse_duration, n_pulses, t, wait_time):
        k = bandwidth / pulse_duration
        pulses = []
        wait = torch.zeros((wait_time))
        # print("DEBUG: ", n_pulses, pulse_duration)
        for n in range(n_pulses):
            pulse = torch.exp(1j * pi * k * torch.pow(t - pulse_duration / 2, 2))
            pulses.append(torch.cat([pulse, wait]))
        lfm = torch.cat(pulses).reshape(n_pulses, -1)
        return lfm

    def doppler_processing(self, signal, data, nfft_range, nfft_doppler):
        range_scaling = torch.sqrt(torch.Tensor([nfft_range]))
        doppler_scaling = torch.tensor(data.shape[-2], dtype=torch.complex64)

        kernel = torch.conj(torch.fft.fft(signal, n=nfft_range, dim=-1, norm='ortho'))

        rows = range_scaling * torch.fft.ifft(
            torch.multiply(kernel, torch.fft.fft(data, n=nfft_range, dim=-1, norm='ortho')), n=nfft_range, dim=-1,
            norm='ortho')

        columns = doppler_scaling * torch.fft.fft(rows, n=nfft_doppler, dim=-2, norm='ortho')

        return torch.fft.fftshift(torch.fft.fftshift(columns, dim=-2), dim=-1)

    def make_scene(self, amplitudes, ranges, velocities, max_unamb_range, max_unamb_vel, nfft_range, nfft_doppler):

        # compute normalized aparent ranges and velocities (considering ambiguos range/vel)
        ranges = torch.fmod(ranges, max_unamb_range) / max_unamb_range
        velocities = torch.fmod(velocities, max_unamb_vel) / max_unamb_vel

        # compute binnned range and velocities
        ranges = torch.round((ranges + 1) * (nfft_range // 2)).long().reshape(1, -1)
        velocities = torch.round((velocities + 1) * (nfft_doppler // 2)).long().reshape(1, -1)

        # fill scene matrix
        scene = torch.zeros(1, nfft_doppler, nfft_range, dtype=torch.complex64)
        for i, (rind, vind) in enumerate(zip(ranges, velocities)):
            scene[0, vind, rind] = amplitudes[i]

        return scene

    def fft(self, signal, nfft: int = None, dim: int = -1):
        data = signal.data
        nfft = nfft if nfft else 2 * self.num_samples - 1
        return torch.fft.fft(input=data, n=nfft, dim=dim, norm='ortho')

    def ifft(self, signal, nfft: int = None, dim: int = -1):
        data = signal.data
        nfft = nfft if nfft else 2 * self.num_samples - 1
        return torch.fft.ifft(input=data, n=nfft, dim=dim, norm='ortho')

    def simulate_target_with_scene_profile(self, signal, scene_profile, num_pulses):

        # scene parameters
        nfft_doppler = scene_profile.shape[-2]
        nfft_range = scene_profile.shape[-1]

        # scaling for correcting signal amplitude
        range_fft_scaling = torch.sqrt(torch.Tensor([nfft_range]))
        doppler_fft_scaling = torch.sqrt(torch.Tensor([nfft_doppler]))

        # kernel from operation
        kernel = self.fft(signal, nfft=nfft_range)

        # arrange Doppler bins
        scene_profile = torch.fft.fftshift(torch.fft.fftshift(scene_profile, dim=-2), dim=-1)

        # propagate dopplers from targets and select the number of pulses
        scene_profile = self.ifft(scene_profile, nfft=nfft_doppler, dim=-2)[:, :num_pulses, :] * doppler_fft_scaling
        # compute fft of the range profiles of each of the pulses
        scene_profile = self.fft(scene_profile, nfft=nfft_range, dim=-1) * range_fft_scaling
        # apply convolution with signal kernel and go back to time domain
        scene_profile = self.ifft(torch.multiply(kernel, scene_profile), nfft=nfft_range, dim=-1)

        return scene_profile

    def get_max_unambigous(self):
        return self.max_unamb_range, self.max_unamb_velocity
