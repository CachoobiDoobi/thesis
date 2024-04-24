import logging

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplex
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from torch_geometric.nn import GCNConv, global_mean_pool
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
        # This does nothing. It would be called in the forward method if this was an agent
        ##################
        self.model = TorchComplex(obs_space, action_space, num_outputs, model_config, name).to(device)
        #############################3
        self.dropout_rate = 0.2  # dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False).to(device)
        self.relu = nn.PReLU().to(device)
        # hardcode number of parameters
        self.embedding = Linear(in_features=6, out_features=6).to(device)
        # self.embedding.to(device)
        self.out = Linear(in_features=6, out_features=1).to(device)
        # self.out.to(device)
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 4 + 6 # equal to action space + EMBEDDINGS
        hidden_dim = 128

        # TODO try with more convs and some varying hidden dim size
        self.convs = ModuleList([
            GCNConv(input_size, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, 1)
        ]).to(device)
        # self.convs.to(device)

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
        embeddings = self.embedding(observation)
        # print(bursts.shape, embeddings.shape)
        # forward
        for batch in loader:
            batch = batch.to(self.device)
            # should always be 1 batch
            x = batch.x
            x = x.to(self.device)
            embeddings = embeddings.unsqueeze(2).expand(-1, -1, 6).reshape(-1, 6)
            x = torch.cat([x, embeddings], dim=1).to(self.device)
            edge_index = batch.edge_index
            edge_weights = batch.edge_attr
            for i in range(len(self.convs) - 1):
                x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weights)
                x = self.relu(x)
                x = self.dropout(x)
            x = self.convs[-1](x=x, edge_index=edge_index)
            # use a FC net here
            x = global_mean_pool(x, batch.batch)
            # x = x.reshape(bursts.shape[0], -1)
#            x = self.out(x)
            return x.reshape(-1)

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
