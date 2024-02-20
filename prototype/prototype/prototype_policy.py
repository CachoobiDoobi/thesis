import logging
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from ray.rllib import Policy
from torch import TensorType, nn


class PrototypePolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

        self.model = nn.Sequential(nn.Linear(observation_space.shape[0], 10), nn.Sigmoid(), nn.Linear(10, 3))

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.loss = nn.MSELoss()

        self.action_space = action_space

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
        print(obs_batch)
        # Step 1: Generate bursts, dxB
        obs_batch = torch.as_tensor(obs_batch)
        # why is this 2x24??? even worse, why does this wwork sometimes
        actions = self.model(obs_batch)

        # TODO map output to action space
        return {param: actions[:, i] for i, param in enumerate(self.action_space.keys())}, [], {}

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
