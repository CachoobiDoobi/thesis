import logging
import math
import platform

import numpy as np
from carpet import carpet
import plotly.graph_objs as go
from config import param_dict


class CarpetSimulation:
    def __init__(self):
        if platform.system() == 'Linux':
            # carpet.read_license("Przlwkofsky")
            carpet.read_license("/project/carpet3.lcs")
        # Processing
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        # what is this?
        carpet.Processing_M = 3


    def detect(self, action_dict, range_, velocity, altitude, wind_speed, rcs, rainfall_rate):
        carpet.Target_Azimuth = 0
        carpet.Processing_DFB = True
        carpet.Processing_MTI = 'no'
        carpet.Processing_Integrator = 'm out n'
        carpet.Clutter_SurfaceClutter = True
        carpet.Target_RCS1 = rcs
        carpet.Propagation_WindDirection = np.pi
        carpet.Propagation_Vwind = wind_speed
        carpet.Clutter_RainPresent = True
        carpet.Clutter_RainfallRate = rainfall_rate
        carpet.Clutter_RainRange = 0
        carpet.Clutter_RainDiameter = 1e6

        for m, agent in enumerate(action_dict):
            parameters = action_dict[agent]
            pulse_durations = parameters.get("pulse_duration")
            n_pulseses = parameters.get('n_pulses')
            pris = parameters.get('PRI')
            n_bursts = len(pris)
            rfs = parameters.get('RF')

            for n in range(1, n_bursts + 1):
                i = str(n + m * n_bursts)
                setattr(carpet, f"Transmitter_PRF{i}", 1 / param_dict["PRI"][pris[n - 1]])
                setattr(carpet, f"Transmitter_Tau{i}", param_dict["pulse_duration"][pulse_durations[n - 1]])
                setattr(carpet, f"Transmitter_PulsesPerBurst{i}", int(n_pulseses[n - 1]))
                setattr(carpet, f"Transmitter_RF{i}", int(param_dict['RF'][rfs[n - 1]]))
                setattr(carpet, f"Transmitter_RF{i}", int(param_dict['RF'][rfs[n - 1]]))
                setattr(carpet, f"Transmitter_NrFilters{i}", int(n_pulseses[n - 1]))
                # carpet.Transmitter_NrFilters

        assert range_ > 0, "NEGATIVE RANGE"

        # carpet.Target_GroundRange = range_
        # carpet.Target_RadialVelocity = velocity
        # carpet.Target_Altitude = altitude

        # pds = carpet.detection_probability(ground_ranges=range_, radial_velocities=velocity, altitudes=altitude)
        # scnr = carpet.signal_clutter_noise_power_ratio(ground_ranges=range_, radial_velocities=velocity,altitudes=altitude)

        data = carpet.detection_probability(ground_ranges=np.linspace(start=1e4, stop=5e4, num=1000),
                                            radial_velocities=np.linspace(start=100, stop=500, num=500), altitudes=altitude)

        carpet.save_config("carpet_radu")
        print(data.shape)
        # # Create a heatmap trace
        heatmap = go.Heatmap(z=data,x=np.linspace(start=1e4, stop=5e4, num=1000),y=np.linspace(start=100, stop=500, num=500), zmin=0, zmax=1)

        layout = go.Layout(
            title='PD',
            xaxis=dict(title='Range'),
            yaxis=dict(title='Velocity')
        )


        # Create figure
        fig1 = go.Figure(data=heatmap, layout=layout)

        fig1.add_trace(go.Scatter(x=[range_], y=[velocity], ))
        fig1.show()


actions = np.load("waveforms.txt.npy", allow_pickle=True)
print(actions[2][0])
ranges = np.loadtxt("ranges.txt")
velocities = np.loadtxt("velocities.txt")
alts = np.loadtxt("alts.txt")

pulse_durations = [param_dict['pulse_duration'][pd] for pd in actions[2][0]["pulse_duration"]]
pris = [param_dict['PRI'][pri] for pri in actions[2][0]["PRI"]]
n_pulses = actions[2][0]['n_pulses']
rfs = [param_dict['RF'][rf] for rf in actions[2][0]["RF"]]

print(f"Pulse Durations: {pulse_durations}")
print(f"Number of pulses: {n_pulses}")
print(f"PRI: {pris}")
print(f"RF: {rfs}")

for i in range(0, 20):
    print(actions[i][0])
    sim = CarpetSimulation()
    sim.detect(actions[i], range_=ranges[i], velocity=velocities[i], altitude=alts[i], wind_speed=18, rcs=1, rainfall_rate=(2.7 * 10e-7)/25)
