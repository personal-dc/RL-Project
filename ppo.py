import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO Hyperparameters
HYPERPARAMS = {
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "lr": 3e-4,
    "update_epochs": 10,
    "batch_size": 64,
    "total_timesteps": 2_000_000,
    "rollout_steps": 2048
}

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.common_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 9, 512),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softplus()  # Alpha, Beta for Beta distribution
        )

        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Channel-last to channel-first
        x = self.common_cnn(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.model = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=HYPERPARAMS["lr"])
        self.memory = []

    def save_model(self, path="ppo_car_racing.pkl"):
        """Save the model's state dictionary."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="ppo_car_racing.pkl"):
        """Load the model's state dictionary."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            params, _ = self.model(state)
        alpha = params[:, 0] + 1
        beta = params[:, 1] + 1
        dist = Beta(alpha, beta)
        action = dist.sample().cpu().numpy().squeeze()
        log_prob = dist.log_prob(torch.tensor(action).to(device)).sum()
        return action, log_prob.item()

    def compute_returns_and_advantages(self, rewards, dones, values):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + HYPERPARAMS["gamma"] * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + HYPERPARAMS["gamma"] * HYPERPARAMS["lambda"] * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self):
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)
        self.memory = []  # Clear memory
        advantages, returns = self.compute_returns_and_advantages(rewards, dones, values)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(device)

        for _ in range(HYPERPARAMS["update_epochs"]):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), HYPERPARAMS["batch_size"]):
                end = start + HYPERPARAMS["batch_size"]
                idx = indices[start:end]

                # Compute loss
                params, value = self.model(states[idx])
                alpha = params[:, 0] + 1
                beta = params[:, 1] + 1
                dist = Beta(alpha, beta)
                log_probs = dist.log_prob(actions[idx]).sum(dim=1)
                ratio = torch.exp(log_probs - old_log_probs[idx])

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - HYPERPARAMS["clip_epsilon"], 1 + HYPERPARAMS["clip_epsilon"]) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = HYPERPARAMS["value_loss_coeff"] * (returns[idx] - value.squeeze()).pow(2).mean()
                entropy_loss = HYPERPARAMS["entropy_coeff"] * dist.entropy().mean()

                loss = actor_loss + critic_loss - entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, total_timesteps):
        state, _ = self.env.reset()
        score_history = []
        ep_rewards = 0
        ep_number = 1

        for t in range(1, total_timesteps + 1):
            action, log_prob = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            ep_rewards += reward
            self.memory.append((state, action, log_prob, reward, done, self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))[1].item()))

            state = next_state

            if done or truncated:
                score_history.append(ep_rewards)
                print(f'Episode: {ep_number} ...    Score : {ep_rewards}')
                ep_rewards = 0
                ep_number+=1
                state, _ = self.env.reset()

            if t % HYPERPARAMS["rollout_steps"] == 0:
                self.update()
                self.save_model("ppo_car_racing.pth")

            if t % (HYPERPARAMS["rollout_steps"] * 10) == 0:
                print(f"Timestep {t}, Average Reward: {np.mean(score_history[-10:])}")

        self.env.close()

# Main
if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    agent = PPOAgent(env)
    agent.train(HYPERPARAMS["total_timesteps"])
