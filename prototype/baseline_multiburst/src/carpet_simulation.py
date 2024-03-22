####### our case #######
import numpy as np
from carpet import carpet
from torch import Tensor

from config import param_dict


class CarpetSimulation:
    def __init__(self, ranges: [Tensor], velocities: [Tensor]):
        self.ranges = ranges
        self.velocities = velocities
        self.doppler_resolution = 0
        self.snr = None

    def detect(self, parameters):
        c = 299792458
        alt = [1000] * self.ranges.shape[0]
        # param_names = ['PRF', 'Tau', 'PulsesPerBurst', 'NrBursts']
        # filtered_attributes = [attr_name for attr_name in dir(carpet) if any(substring in attr_name for substring in param_names)]

        pulse_durations = parameters.get("pulse_duration")
        n_pulseses = parameters.get('n_pulses')
        pris = parameters.get('PRI')
        n_bursts = len(pris)

        for n in range(1, n_bursts + 1):
            i = str(n)
            setattr(carpet, f"Transmitter_PRF{i}", 1 / param_dict["PRI"][pris[n - 1]])
            setattr(carpet, f"Transmitter_Tau{i}", param_dict["pulse_duration"][pulse_durations[n - 1]])
            setattr(carpet, f"Transmitter_PulsesPerBurst{i}", int(n_pulseses[n - 1]))

        carpet.Clutter_SurfaceClutter = True
        carpet.Propagation_WindDirection = np.pi
        # carpet.Surface

        # extra Doppler filter banks processing and moving target indication # according to Matijs this Doppler filter banks needs to be turned on? Not sure what it does.
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        # what is this
        carpet.Processing_M = 3
        pds = carpet.detection_probability(ranges=self.ranges, velocities=self.velocities, heights=alt)
        return pds.reshape(-1)[0]
