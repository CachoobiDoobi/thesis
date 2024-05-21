import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


path = ""

fig = make_subplots(2, 3)
fig.add_trace(go.Histogram2d(
    x = np.loadtxt("single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 1, 1)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 1, 2)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 1, 3)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 2, 1)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 2, 2)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),
    coloraxis = "coloraxis",), 2, 3)

fig.show()