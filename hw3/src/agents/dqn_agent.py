from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        
        Args:
            observation: (observation_shape, ) nparray
        Returns:
            action: int
        """
        observation = ptu.from_numpy(np.asarray(observation))[None] # [1, observation_shape] torch.Tensor

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        if torch.rand(1).item() <= epsilon: # pick a random action
            action = (self.num_actions * torch.rand(1)).floor() # [1] torch.Tensor
        else:
            # select the most likely action
            actions = self.critic(observation) # [1, num_actions] torch.Tensor
            _, action = torch.max(input=actions, dim=1) # [1] torch.Tensor, which is our index!
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,      # [B, *observation_shape]
        action: torch.Tensor,   # [B, ]
        reward: torch.Tensor,   # [B, ]
        next_obs: torch.Tensor, # [B, *observation_shape]
        done: torch.Tensor,     # [B, ]
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values
            next_qa_values = self.target_critic(next_obs) # [B, num_actions] torch.Tensor
            # within a batch, we have a num_actions representing the logits for each action
            
            if self.use_double_q:
                # TODO(Section 2.5): implement double-Q target action selection
                next_action = self.critic(next_obs).argmax(dim=1)
            else:
                next_action = next_qa_values.argmax(dim=1) # [B, ]
                # math notation: a'

            next_q_values = next_qa_values[torch.arange(batch_size), next_action] # [B, ] torch.Tensor
            # these are the Q values if we take next_action, or Q(s', a')
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount * next_q_values * (1.0 - done.float())
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = qa_values[torch.arange(batch_size), action] # Q(s, a)
        loss = torch.nn.functional.mse_loss(input=q_values, target=target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )
        # Hint: if step % self.target_update_period == 0: ...
        # if this is true, we need to update the target w/ the weights of the critic
        if step % self.target_update_period == 0:
            self.update_target_critic()
        # ENDTODO

        return critic_stats
