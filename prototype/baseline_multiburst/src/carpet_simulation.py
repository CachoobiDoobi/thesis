####### our case #######
import numpy as np
from carpet import carpet
from torch import Tensor

from prototype.baseline.config import param_dict


class CarpetSimulation:
    def __init__(self, ranges: [Tensor], velocities: [Tensor]):
        self.ranges = ranges
        self.velocities = velocities
        self.doppler_resolution = 0
        self.snr = None

    def detect(self, parameters):
        c = 299792458
        alt = [1000] * self.ranges.shape[0]
        param_names = ['PRF', 'Tau', 'PulsesPerBurst', 'NrBursts']
        filtered_attributes = [attr_name for attr_name in dir(carpet) if any(substring in attr_name for substring in param_names)]

        pulse_durations = parameters.get("pulse_duration")
        n_pulseses = parameters.get('n_pulses')
        pris = parameters.get('PRI')
        n_bursts = len(pris)

        for n in range(n_bursts):
            i = str(n)
            for attr_name in filtered_attributes:
                if i in attr_name and 'PRF' in attr_name:
                    setattr(carpet, attr_name, 1 / param_dict["PRI"][pris[n]])
                elif i in attr_name and 'Tau' in attr_name:
                    setattr(carpet, attr_name, param_dict["pulse_duration"][pulse_durations[n]])
                elif i in attr_name and 'PulsesPerBurst' in attr_name:
                    setattr(carpet, attr_name, int(n_pulseses[n]))


        # transmit burst 1
        # carpet.Transmitter_PRF1 = 2000  # Hz
        # carpet.Transmitter_PulsesPerBurst1 = 17
        # carpet.Transmitter_NrBursts1 = 1
        # carpet.Transmitter_Tau1 = 10e-06
        # # carpet.Transmitter_RF1 = 25e9
        #
        # # transmit burst 2 (not working properly. Update 2024-02-08: Fixed, you need the licensed version of python carpet -- Ask Arne)
        # carpet.Transmitter_PRF2 = 1300  # Hz
        # carpet.Transmitter_PulsesPerBurst2 = 17
        # carpet.Transmitter_NrBursts2 = 1
        # carpet.Transmitter_Tau2 = 10e-06
        #
        # # transmit burst 3
        # carpet.Transmitter_PRF3 = 700  # Hz
        # carpet.Transmitter_PulsesPerBurst3 = 17
        # carpet.Transmitter_NrBursts3 = 1
        # carpet.Transmitter_Tau3 = 10e-06

        # extra Doppler filter banks processing and moving target indication # according to Matijs this Doppler filter banks needs to be turned on? Not sure what it does.
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        # what is this
        carpet.Processing_M = 3
        pds = carpet.detection_probability(ranges=self.ranges, velocities=self.velocities, heights=alt)
        return pds.reshape(-1)[0]

