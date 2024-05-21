import networkx as nx
import numpy as np
import torch
from numpy.linalg import norm
from ray.rllib.models.modelv2 import restore_original_dimensions
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import plotly.graph_objs as go

import plotly.io as pio

rcs = np.linspace(1, 20, num=20)
wind_speed = np.linspace(start=0, stop=40, num=20)
rainfall_rate = np.linspace(start=0, stop=2.8 * 1e-6, num=20)

def preprocess_observations(obs, opponent_obs, original_obs_space):
    original_obs = restore_original_dimensions(obs=obs, obs_space=original_obs_space, tensorlib="torch")
    original_opponent_obs = restore_original_dimensions(obs=opponent_obs, obs_space=original_obs_space,
                                                        tensorlib="torch")
    # reverse one hot encoding
    batch_size = original_obs['PRI'].shape[0]
    for key in original_obs:
        original_obs[key] = original_obs[key].reshape(batch_size, original_obs_space[key].shape[0], -1)
        original_opponent_obs[key] = original_opponent_obs[key].reshape(batch_size,
                                                                        original_obs_space[key].shape[0], -1)
    original_obs = {key: torch.argmax(original_obs[key], dim=-1) for key in original_obs}
    original_opponent_obs = {key: torch.argmax(original_opponent_obs[key], dim=-1) for key in original_opponent_obs}
    # extract burst params
    n_bursts = original_obs['PRI'].shape[1] + original_opponent_obs['PRI'].shape[1]
    bursts = torch.zeros(batch_size, n_bursts, 4)  # hardcoded :(
    # burst is [batch size, n_bursts, params], obs is keys, batch size, params
    observation = torch.zeros(batch_size, 6)
    for n in range(n_bursts):
        i = 0
        j = 0
        for key in original_obs:
            if key in ['PRI', 'n_pulses', 'pulse_duration', 'RF']:
                if n < n_bursts // 2:
                    bursts[:, n, i] = original_obs[key][:, n]
                else:
                    bursts[:, n, i] = original_opponent_obs[key][:, n - n_bursts // 2]
                i += 1
            else:
                observation[:, j] = original_obs[key].reshape(-1)
                j += 1
    return bursts, observation


def build_graphs_from_batch(bursts):
    # TODO build radar-informed graph
    # Can make edge features for each param,
    # Edge feature can be based on actual blind ranges or other things
    batch_size, n_bursts, _ = bursts.shape

    # Function to calculate dissimilarity between bursts
    def calculate_similarity(burst1, burst2):
        # print(burst1, burst2)
        burst1 = np.array([burst1[0], burst1[1], burst1[3]])
        burst2 = np.array([burst2[0], burst2[1], burst2[3]])
        # print(f'bursts: {burst1, burst2}')
        if np.any(burst1) and np.any(burst2):
            # cosine similarity
            return np.dot(burst1, burst2) / (norm(burst1) * norm(burst2))
        return 0.5

    # nodes = []
    # edges = []
    data_list = []
    for i in range(batch_size):
        graph = nx.Graph()
        # Add nodes (bursts) with their parameters as node attributes
        for j in range(n_bursts):
            param = bursts[i, j].tolist()  # Convert torch tensor to tuple
            graph.add_node(j, param=param)

        disim_matrix = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
        # Add edges based on dissimilarity
        for i, node1 in enumerate(graph.nodes()):
            for j, node2 in enumerate(graph.nodes()):
                if node1 != node2:  # No self-loops
                    similarity = calculate_similarity(graph.nodes[node1]['param'], graph.nodes[node2]['param'])
                    disim_matrix[i, j] = 1 - similarity
        disim_matrix = disim_matrix / np.max(disim_matrix)
        for i, node1 in enumerate(graph.nodes()):
            for j, node2 in enumerate(graph.nodes()):
                if node1 != node2 and 0.5 <= disim_matrix[i, j]:
                    graph.add_edge(node1, node2, weight=disim_matrix[i, j])
        node_features = torch.tensor([graph.nodes[node]['param'] for node in graph.nodes()])
        edge_indices = torch.tensor(list(graph.edges()), dtype=torch.int64).t().contiguous()

        edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)
        print(f'The number of edges is: {graph.number_of_edges()}, edge weights: {edge_weights_tensor}')
        data = Data(x=node_features,
                    edge_index=edge_indices, edge_attr=edge_weights_tensor)
        data_list.append(data)

    return DataLoader(data_list, batch_size=batch_size,
                      shuffle=True)  # torch.cat(nodes, dim=0), torch.cat(edges, dim=0) #Batch.from_data_list(data_list)

def plot_heatmaps_rcs_wind(pds, ratios, track):
    # Create a heatmap trace
    heatmap = go.Heatmap(z=pds, x=wind_speed, y=rcs, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='PD',
        xaxis=dict(title='Wind Speed'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig1 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig1, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_wind_pd.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=ratios, x=wind_speed, y=rcs, zmin=3, zmax=7)

    layout = go.Layout(
        title='Waveform Duration Ratio',
        xaxis=dict(title='Wind Speed'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig2 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig2, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_wind_ratios.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=track, x=wind_speed, y=rcs, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='Firm Track Probability',
        xaxis=dict(title='Wind Speed'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig3 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig3, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_wind_track.pdf')



def plot_heatmaps_rcs_rainfall(pds, ratios, track):
    # Create a heatmap trace
    heatmap = go.Heatmap(z=pds, x=rainfall_rate, y=rcs, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='PD',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig1 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig1, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_rain_pd.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=ratios, x=rainfall_rate, y=rcs, zmin=3, zmax=7)

    layout = go.Layout(
        title='Waveform Duration Ratio',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig2 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig2, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_rain_ratios.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=track, x=rainfall_rate, y=rcs, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='Firm Track Probability',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Radar Cross Section')
    )

    # Create figure
    fig3 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig3, '/project/multi_agent_critic_gnn_threshold/results/heatmap_rcs_rain_track.pdf')


def plot_heatmaps_wind_rainfall(pds, ratios, track):
    # Create a heatmap trace
    heatmap = go.Heatmap(z=pds, x=rainfall_rate, y=wind_speed, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='PD',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Wind Speed')
    )

    # Create figure
    fig1 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig1, '/project/multi_agent_critic_gnn_threshold/results/heatmap_wind_rain_pd.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=ratios, x=rainfall_rate, y=wind_speed, zmin=3, zmax=7)

    layout = go.Layout(
        title='Waveform Duration Ratio',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Wind Speed')
    )

    # Create figure
    fig2 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig2, '/project/multi_agent_critic_gnn_threshold/results/heatmap_wind_rain_ratios.pdf')

    # Create a heatmap trace
    heatmap = go.Heatmap(z=track, x=rainfall_rate, y=wind_speed, zmin=0.6, zmax=1)

    layout = go.Layout(
        title='Firm Track Probability',
        xaxis=dict(title='Rainfall Rate'),
        yaxis=dict(title='Wind Speed')
    )

    # Create figure
    fig3 = go.Figure(data=heatmap, layout=layout)

    # Save the first plot to a file
    pio.write_image(fig3, '/project/multi_agent_critic_gnn_threshold/results/heatmap_wind_rain_track.pdf')


def plot_2d_hist(track, ratios):

    track = np.array(track).reshape(-1)
    track = np.round(track, decimals=2)
    ratios = np.array(ratios).reshape(-1)
    ratios = np.round(ratios, decimals=2)

    iqr = np.percentile(track, 75) - np.percentile(track, 25)
    bin_width = 2 * iqr / (len(track) ** (1 / 3))  # Freedman-Diaconis rule
    num_bins_x = int(np.ceil((np.max(track) - np.min(track)) / bin_width))

    iqr = np.percentile(ratios, 75) - np.percentile(ratios, 25)
    bin_width = 2 * iqr / (len(ratios) ** (1 / 3))  # Freedman-Diaconis rule
    num_bins_y = int(np.ceil((np.max(ratios) - np.min(ratios)) / bin_width))

    layout = go.Layout(
        title='Waveform distribution',
        xaxis=dict(title='Firm track probability'),
        yaxis=dict(title='Waveform duration ratio')
    )
    # Create the 2D histogram
    fig = go.Figure(data=go.Histogram2d(x=track, y=ratios, nbinsx=num_bins_x, nbinsy=num_bins_y), layout=layout)

    pio.write_image(fig, '/project/multi_agent_critic_gnn_threshold/results/2dhist.pdf')