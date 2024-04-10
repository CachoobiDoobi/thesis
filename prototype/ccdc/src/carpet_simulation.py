####### our case #######
import math

import numpy as np
from carpet import carpet
from torch import Tensor

from config import param_dict


# TODO do the Arne things
class CarpetSimulation:
    def __init__(self, ranges: [Tensor], velocities: [Tensor]):
        self.ranges = ranges
        self.velocities = velocities
        self.altitudes = np.random.uniform(low=10, high=1e4, size=self.ranges.shape[0])
        self.doppler_resolution = 0
        self.snr = None

        carpet.Clutter_SurfaceClutter = True
        carpet.Target_RCS1 = np.random.uniform(2, 20)
        carpet.Propagation_WindDirection = np.pi
        carpet.Propagation_Vwind = np.random.uniform(20, 30)
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        # what is this?
        carpet.Processing_M = 3

    def detect(self, action_dict):

        for m, agent in enumerate(action_dict):
            parameters = action_dict[agent]
            pulse_durations = parameters.get("pulse_duration")
            n_pulseses = parameters.get('n_pulses')
            pris = parameters.get('PRI')
            n_bursts = len(pris)

            for n in range(1, n_bursts + 1):
                i = str(n + m * n_bursts)
                setattr(carpet, f"Transmitter_PRF{i}", 1 / param_dict["PRI"][pris[n - 1]])
                setattr(carpet, f"Transmitter_Tau{i}", param_dict["pulse_duration"][pulse_durations[n - 1]])
                setattr(carpet, f"Transmitter_PulsesPerBurst{i}", int(n_pulseses[n - 1]))

        r = float(self.ranges[0])
        vel = float(self.velocities[0])
        alt = float(self.altitudes[0])
        # print(r, vel , alt)
        pds = carpet.detection_probability(ground_ranges=r, radial_velocities=vel, altitudes=alt)
        scnr = carpet.GetSCNR()
        # print(pds, scnr)
        return (pds, scnr) if not math.isnan(pds) else (0, 0)
