import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from self_train import WrapperEnv

def train_regular(timesteps = 1000000):
    # env = DiscretizedCarRacing()
    env = gym.make('CarRacing-v3', render_mode='rgb_array')


    # Wrap the environment with DummyVecEnv (Stable-Baselines3 requires vectorized environments)
    env = DummyVecEnv([lambda: env])
    print(env.observation_space)

    # Define PPO model
    model = PPO('CnnPolicy', env, verbose=1)
    # model = DQN('CnnPolicy', env, verbose=1)


    # Train the model
    model.learn(total_timesteps=timesteps)  # You can increase the timesteps for better training

    # Save the model
    model.save("ppo_carracing_regular_env")

# # Evaluate the model
# obs = env.reset()
# for _ in range(1000):  # Test for 1000 steps (you can adjust based on your needs)
#     action, _states = model.predict(obs)
#     obs, rewards, dones, infos = env.step(action)
#     env.render()

# # Close the environment after testing
# env.close()
