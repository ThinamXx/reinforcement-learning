from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

plt.rcParams["figure.figsize"] = (12, 8)


class PolicyNetwork(nn.Module):
    """Parameterized policy network"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        """Initialize the neural network that estimates the mean and standard deviation
        of a normal distribution from which the actions are sampled from.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in the neural network.
        """
        super().__init__()

        # Shared network
        self.shared_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy mean specific network
        self.policy_mean_network = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Policy standard deviation specific network
        self.policy_std_network = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, return the mean and standard deviation of the normal distribution.

        Args:
            x (torch.Tensor): Observation from the environment.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the normal distribution.
        """
        shared_features = self.shared_network(x.float())
        action_mean = self.policy_mean_network(shared_features)
        action_std = torch.log(1 + torch.exp(self.policy_std_network(shared_features)))

        return action_mean, action_std


class REINFORCE:
    """REINFORCE algorithm"""

    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize an agent that uses the REINFORCE algorithm to learn a policy.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
        """

        # Hyperparameters
        self.lr = 1e-4  # Learning rate for the policy optimization.
        self.gamma = 0.99  # Discount factor.
        self.eps = 1e-6  # Small value for numerical stability.

        self.probs = []  # Stores the probabilities of the actions taken.
        self.rewards = []  # Stores the rewards obtained.

        self.net = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

    def sample_action(self, state: np.ndarray) -> float:
        """Return an action conditioned on the policy and observation.

        Args:
            state (np.ndarray): Observation from the environment.

        Returns:
            action: Action sampled from the policy network to take in the environment.
        """
        state = torch.tensor(np.array([state]))
        action_mean, action_std = self.net(state)

        # Create a normal distribution from mean and standard deviation.
        dist = Normal(action_mean[0] + self.eps, action_std[0] + self.eps)
        action = dist.sample()

        # Store the probability of the action taken.
        self.probs.append(dist.log_prob(action))

        return action.detach().numpy()

    def update(self):
        """Update the policy network using the REINFORCE algorithm."""

        # Compute the discounted rewards.
        discounted_rewards = []
        cumulative_reward = 0
        for reward in self.rewards[::-1]:
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards)

        # Normalize the discounted rewards.
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + self.eps
        )

        # Compute the loss.
        loss = 0
        for log_prob, discounted_reward in zip(self.probs, discounted_rewards):
            loss += -log_prob.mean() * discounted_reward

        # Update the policy network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset the memory.
        self.probs = []
        self.rewards = []

        return loss.item()


def train():
    # Create and wrap the environment.
    env = gym.make("InvertedPendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(
        env, 50
    )  # Record the episode statistics.

    total_episodes = 50000  # Total number of episodes to train the agent.
    obs_dim = env.observation_space.shape[0]  # Dimension of the observation space.
    action_dim = env.action_space.shape[0]  # Dimension of the action space.

    reward_over_seeds = []  # Store the reward obtained for each seed.

    for seed in [1, 2, 3, 5, 8]:
        # Set the seed for reproducibility.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize the agent.
        agent = REINFORCE(obs_dim, action_dim)
        reward_over_episodes = []  # Store the reward obtained for each episode.

        for episode in range(total_episodes):
            obs, info = wrapped_env.reset(
                seed=seed
            )  # Reset the environment and record the initial state.

            done = False
            while not done:
                action = agent.sample_action(obs)

                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)

                # End the episode if the environment is terminated or truncated.
                # terminated: The episode completed the state space.
                # truncated: The episode duration reached the maximum limit.
                done = terminated or truncated

            reward_over_episodes.append(
                wrapped_env.return_queue[-1]
            )  # Record the reward obtained for the episode.
            agent.update()  # Update the policy network.

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print(f"Episode: {episode}, Reward: {avg_reward}")

    reward_over_seeds.append(reward_over_episodes)


if __name__ == "__main__":
    train()
    print("Training complete!!")
