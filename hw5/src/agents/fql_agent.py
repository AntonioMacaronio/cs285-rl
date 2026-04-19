from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None] # [1, obs_dim]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        noise = torch.randn((1, self.action_dim), device=observation.device) # [1, action_dim]
        action = self.onestep_actor(observation, noise) # [1, action_dim]
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        dt =  1.0 / self.flow_steps
        action = noise
        for i in range(self.flow_steps):
            t = torch.full((*action.shape[:-1], 1), i * dt, device=action.device)
            direction = self.bc_actor(observation, action, t) # [B, action_dim]
            action = action + dt * direction
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        B = actions.shape[0]
        noise = torch.randn((B, self.action_dim), device=observations.device) # [B, action_dim]
        q = self.critic(observations, actions) # [2, B]

        with torch.no_grad():
            next_actions = torch.clamp(self.onestep_actor(next_observations, noise), min=-1, max=1)
            target = rewards + self.discount * (1 - dones.float()) * self.target_critic(next_observations, next_actions).mean(dim=0)

        loss = ((target - q)**2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        B = actions.shape[0]
        t = torch.rand((B, 1), device=actions.device)
        noise = torch.randn((B, self.action_dim), device=actions.device)
        noisy_action = (1-t)*noise + t*actions
        velocity = self.bc_actor(observations, noisy_action, t) # [B, action_dim]
        loss = ((velocity - (actions - noise)) ** 2).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        B, action_dim = actions.shape
        noise = torch.randn((B, action_dim), device=observations.device)
        onestep_action = self.onestep_actor(observations, noise) # [B, action_dim]
        with torch.no_grad():
            flow_action = self.get_bc_action(observations, noise) # [B, action_dim]
        distill_loss = self.alpha * ((onestep_action - flow_action)**2).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        onestep_action_clipped = torch.clamp(onestep_action, min=-1, max=1)
        q_loss = -self.critic(observations, onestep_action_clipped).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        with torch.no_grad():
            mse = ((onestep_action - actions) ** 2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
          for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
              p_target.data.lerp_(p.data, self.target_update_rate)
