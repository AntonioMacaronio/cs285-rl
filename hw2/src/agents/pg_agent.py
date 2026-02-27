from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],          # list of (T_i, ob_dim) arrays
        actions: Sequence[np.ndarray],      # list of (T_i,) or (T_i, ac_dim) arrays
        rewards: Sequence[np.ndarray],      # list of (T_i,) arrays
        terminals: Sequence[np.ndarray],    # list of (T_i,) arrays
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards) # list of per-trajectory Q values

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs_np = np.concatenate(obs)            # [T_0 + T_1 + ..., ob_dim]
        actions_np = np.concatenate(actions)    # [T_0 + T_1 + ..., ac_dim]
        rewards_np = np.concatenate(rewards)    # [T_0 + T_1 + ...,       ]
        q_values_np = np.concatenate(q_values)  # [T_0 + T_1 + ...,       ]
        terminals_np = np.concatenate(terminals)# [T_0 + T_1 + ...,       ],

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs_np, rewards_np, q_values_np, terminals_np
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(
            obs_np,
            actions_np,
            advantages,
        )

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for i in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(
                    obs_np,
                    q_values_np,
                )
                info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        gamma_np = self.gamma * np.ones_like(rewards) # [gamma, gamma, ...] (T times)
        exponents = np.array(list(range(len(rewards)))) # [0, 1, 2, ..., T]
        gamma_powers = np.power(gamma_np, exponents) # [gamma^0, gamma^1, ...]
        return [np.sum(np.multiply(gamma_powers, rewards)).item() for i in range(len(rewards))]

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        T = len(rewards)
        output = np.zeros(T)
        output[-1] = rewards[-1]
        for i in range(T - 2, -1, -1):
            output[i] = rewards[i] + self.gamma * output[i + 1]
        return list(output)


    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function.
        
        Args:
            rewards - list of per-trajectory arrays
        Returns:
            q_values - list of q_values for each of those trajectories
        """

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,                   # Numpy arrays are the input!
        obs: np.ndarray,        # (T, ob_dim)
        rewards: np.ndarray,    # (T, ), represents reward for each timestep of a trajectory
        q_values: np.ndarray,   # (T, )
        terminals: np.ndarray,  # (T, )
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).squeeze(-1) # (T, ) numpy array
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])         # (T+1, )
                advantages = np.zeros(batch_size + 1)   # (T+1, )

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i] == 1: # last state! we need to reset the advantage function
                        advantages[i] = rewards[i] - values[i]
                    else:
                        advantages[i] = rewards[i] + self.gamma*values[i+1] - values[i] + self.gae_lambda*self.gamma*advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages
