import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


path = ""

layout = go.Layout(
        title='Waveform distribution',
        xaxis=dict(title='Firm track probability'),
        yaxis=dict(title='Waveform duration ratio')
    )

fig = make_subplots(2, 3, subplot_titles=["Single Agent Baseline", "Multi Agent Baseline", "Multi Agent Critic (FC Net)", "Multi Agent Critic (GNN Stochastic)", "Multi Agent Critic (GNN FC)", "Multi Agent Critic (GNN Threshold)"], x_title='Firm track probability', y_title='Waveform duration ratio',)
fig.add_trace(go.Histogram2d(
    x = np.loadtxt("single_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"single_agent_baseline/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 1, 1)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"multi_agent_baseline/results/track.txt"),
    y = np.loadtxt(path+"multi_agent_baseline/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 1, 2)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"multi_agent_critic_fc/results/track.txt"),
    y = np.loadtxt(path+"multi_agent_critic_fc/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 1, 3)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"multi_agent_critic_gnn/results/track.txt"),
    y = np.loadtxt(path+"multi_agent_critic_gnn/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 2, 1)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"multi_agent_critic_gnn_fc_graph/results/track.txt"),
    y = np.loadtxt(path+"multi_agent_critic_gnn_fc_graph/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 2, 2)

fig.add_trace(go.Histogram2d(
    x = np.loadtxt(path+"multi_agent_critic_gnn_threshold/results/track.txt"),
    y = np.loadtxt(path+"multi_agent_critic_gnn_threshold/results/ratios.txt"),zmin=0, zmax=0.3,bingroup=1,histnorm='probability',
    coloraxis = "coloraxis",), 2, 3)



fig.show()