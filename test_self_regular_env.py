import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Beta

'''CNN for Actor-Critic PPO'''
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.common_cnn = nn.Sequential(  # input shape (96, 96, 3)
            nn.Conv2d(3, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        self.value = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self.init_weights)

    def forward(self, state):
            state = state.permute(0, 3, 1, 2)
            state = self.common_cnn(state)
            state = state.view(-1, 256)
            value = self.value(state)
            state = self.fc(state)
            alpha = self.alpha(state) + 1
            beta = self.beta(state) + 1

            return (alpha, beta), value

    # initialize weights so that gradient updates are okay in the beginning
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

# Load the saved model parameters
def load_trained_model(filepath):
    model = CNN().double().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Test the trained policy
def test_policy(env, model, num_episodes=5, render=True):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            if render:
                env.render()

            # Prepare state for the model
            state_tensor = torch.from_numpy(state).double().to(device).unsqueeze(0)

            # Get action from the policy
            with torch.no_grad():
                alpha, beta = model(state_tensor)[0]
                dist = Beta(alpha, beta)
                action = dist.mean  # Use the mean of the distribution as a deterministic action

            # Interact with the environment
            state, reward, done, truncated, _ = env.step(action.squeeze().cpu().numpy() * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            total_reward += reward

            if done or truncated:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                break

    env.close()

if __name__ == "__main__":
    # Set up the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the environment
    env = gym.make("CarRacing-v3", render_mode="human")

    # Load the trained model
    model_path = "param/ppo_net_params.pkl"  # Path to the saved model
    trained_model = load_trained_model(model_path)

    # Test the policy
    test_policy(env, trained_model, num_episodes=5, render=True)