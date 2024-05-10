from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplex
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from torch.nn import Linear

from utils import preprocess_observations, mask_bursts

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
        # hardcode number of parameters
        self.embedding = Linear(in_features=6, out_features=6).to(device)
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 10  # equal to action space + EMBEDDINGS
        hidden_dim = 128

        self.vf = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        # self.convs.to(device)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    # computes the central value function based on the Joint Observations
    def central_value_function(self, obs, opponent_obs, opponent_actions):
        # restore original shape
        bursts, observations, mask = preprocess_observations(obs=obs, opponent_obs=opponent_obs,
                                                      original_obs_space=self.original_obs_space)

        bursts = mask_bursts(bursts, mask)
        output = torch.zeros(mask.shape[0]).to(self.device)
        for i, burst in enumerate(bursts):
            burst = burst.to(self.device)

            observation = observations[i].to(self.device)
            embeddings = self.embedding(observation).expand(burst.shape[0], 6)
            x = torch.cat([burst, embeddings], dim=1).to(self.device)
            out = self.vf(x)
            out = torch.mean(out)
            output[i] = out
        return output

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
