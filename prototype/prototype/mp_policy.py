from typing import List, Optional, Dict, Tuple

import torch
from ray.rllib import Policy
from torch import TensorType, nn
from torch_geometric.nn import MessagePassing

class MessagePassingPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # example parameter
        self.model = nn.ModuleList()
        self.model.append(MessagePassing(aggr="add", flow="source_to_target", node_dim=-2))
        self.optimizer = torch.optim.Adam(self.linear.parameters())

        self.loss = nn.MSELoss()

    def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["Episode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # Step 1: Generate bursts, dxB

        # filter out the measurement data
        actions = self.model(obs_batch)

        return actions.detach().numpy(), [], {}

    def learn_on_batch(self, samples):
        # Convert the batch of experiences to Tensors
        obs_batch = samples["obs"]
        actions_batch = samples["actions"]
        rewards_batch = samples["rewards"]

        # Forward pass: Compute predicted actions by passing observations to the model
        predicted_actions, _, _ = self.compute_actions(obs_batch)
        targets = rewards_batch + self.gamma * predicted_actions

        # Compute loss between the predicted actions and the actual actions
        loss = self.loss_fn(targets, actions_batch)

        # Backward pass: Perform a single optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
