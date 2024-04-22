####### our case #######
import logging
import math
import platform

import numpy as np
from carpet import carpet

from config import param_dict


class CarpetSimulation:
    def __init__(self):
        if platform.system() == 'Linux':
            # carpet.read_license("Przlwkofsky")
            carpet.read_license("/project/carpet3.lcs")
        # Reset
        #carpet.reset_radars()
        # Set conditions
        #carpet.Clutter_SurfaceClutter = True
        #carpet.Target_RCS1 = np.random.uniform(2, 20)
        # carpet.Propagation_WindDirection =
        #carpet.Propagation_Vwind = np.random.uniform(0, 30)

        # Processing
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        # what is this?
        carpet.Processing_M = 3

    def detect(self, action_dict, range_, velocity, altitude, wind_speed, rcs):
        carpet.Target_Azimuth = 0
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        carpet.Clutter_SurfaceClutter = True
        carpet.Target_RCS1 = rcs
        carpet.Propagation_WindDirection = np.pi
        carpet.Propagation_Vwind = wind_speed
        # what is this?
        carpet.Processing_M = 3
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

        assert range_ > 0, "NEGATIVE RANGE"

        carpet.Target_GroundRange = range_
        carpet.Target_RadialVelocity = velocity
        carpet.Target_Altitude = altitude

        pds = carpet.detection_probability(ground_ranges=range_, radial_velocities=velocity, altitudes=altitude)
        scnr = carpet.signal_clutter_noise_power_ratio(ground_ranges=range_, radial_velocities=velocity,altitudes=altitude)

        return (pds, scnr) if not math.isnan(pds) else (0, 0)

    # not thread safe
    # def getWindSpeed(self):
    #     return carpet.Propagation_Vwind
