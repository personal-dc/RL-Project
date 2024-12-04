import gymnasium as gym

env = gym.make('CarRacing-v3', render_mode='human')

print(env.reset()[0].shape)