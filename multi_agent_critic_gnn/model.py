import logging

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplex
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling, GraphNorm
from torch.nn import ModuleList, Linear

from utils import build_graphs_from_batch, preprocess_observations

torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.original_obs_space = obs_space.original_space

        input_size = 4 + 6  # equal to action space + EMBEDDINGS
        hidden_dim = 128

        ##################
        # This does nothing. It would be called in the forward method if this was an agent
        self.model = TorchComplex(obs_space, action_space, num_outputs, model_config, name).to(device)

        #############################
        ########## MISC #############
        #############################
        self.dropout_rate = 0.2  # dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False).to(device)
        self.activation = nn.Sigmoid().to(device)
        self.graph_norm = GraphNorm(hidden_dim).to(device)

        #############################
        ### ENCODER #################
        #############################
        # hardcode number of parameters
        self.encoder_nodes = Linear(in_features=4, out_features=4).to(device)
        self.encoder_obs = Linear(in_features=6, out_features=6).to(device)
        self.encoder_edges = Linear(in_features=1, out_features=1).to(device)

        ##############################
        ### DECODER ##################
        ##############################
        self.decoder1 = torch.nn.Linear(hidden_dim, hidden_dim//2).to(device)
        self.decoder2 = torch.nn.Linear(hidden_dim//2, 1).to(device)

        ##############################
        #### GNN #####################
        ##############################

        # The ratios are hardcoded in such a way that we get 1 node at the end. For a graph of a different size, might need different ratios.
        self.layers = ModuleList([
            GCNConv(input_size, hidden_dim),
            TopKPooling(hidden_dim, ratio=0.7),
            GCNConv(hidden_dim, hidden_dim),
            TopKPooling(hidden_dim, ratio=0.5),
            GCNConv(hidden_dim, hidden_dim),
            TopKPooling(hidden_dim, ratio=0.1),
        ]).to(device)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    # computes the central value function based on the Joint Observations
    def central_value_function(self, obs, opponent_obs, opponent_actions):
        # restore original shape
        bursts, observation = preprocess_observations(obs=obs, opponent_obs=opponent_obs, original_obs_space=self.original_obs_space)
        bursts = bursts.to(self.device)
        observation = observation.to(self.device)

        # build graph here
        loader = build_graphs_from_batch(bursts)
        embeddings = self.encoder_obs(observation)
        # forward
        for batch in loader:
            batch = batch.to(self.device)
            # should always be 1 batch
            x = batch.x
            x = x.to(self.device)

            x = self.encoder_nodes(x)

            embeddings = embeddings.unsqueeze(2).expand(-1, -1, 6).reshape(-1, 6)
            x = torch.cat([x, embeddings], dim=1).to(self.device)

            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)

            edge_weights = self.encoder_edges(edge_weights.reshape(-1, 1)).reshape(-1)

            batch = batch.batch
            batch = batch.to(self.device)

            for layer in self.layers:

                if isinstance(layer, GCNConv):
                    x = layer(x=x, edge_index=edge_index, edge_weight=edge_weights)
                    x = self.activation(x)
                    x = self.dropout(x)
                elif isinstance(layer, TopKPooling):
                    x, edge_index, edge_weights, batch, _, _ = layer(x=x, edge_index=edge_index, edge_attr=edge_weights, batch=batch)

                x = self.graph_norm(x, batch)
            x = self.decoder1(x)
            x = self.decoder2(x)

            return x.reshape(-1)

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
