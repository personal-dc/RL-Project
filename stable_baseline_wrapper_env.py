from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.envs.box2d import CarRacing

# Create the environment
env = CarRacing()

# Wrap the environment for stable baselines
env = DummyVecEnv([lambda: env])

# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# Use the PPO algorithm with the CnnPolicy
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_car_racing")