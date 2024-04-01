from gymnasium.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from fc_model import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplex
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Base of the model
        # TODO torch fc doesnt work
        self.model = TorchComplex(obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = obs_space.shape[0]
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        # TODO is this the correct input?
        # input_ = torch.cat(
        #     [
        #         obs,
        #         opponent_obs,
        #         torch.nn.functional.one_hot(opponent_actions.long(), 2).float(),
        #     ],
        #     1,
        # )
        return torch.reshape(self.central_vf(obs), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used


class YetAnotherTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.action_model = TorchComplex(
            obs_space,  # one-hot encoded Discrete(6)
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )

        self.value_model = TorchFC(
            obs_space, action_space, 1, model_config, name + "_vf"
        )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # Store model-input for possible `value_function()` call.
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        # return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)
        return self.action_model({"obs": input_dict["obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2]
        )
        return torch.reshape(value_out, [-1])
