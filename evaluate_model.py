import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

env = gym.make('CarRacing-v3', render_mode='human')
print(env.spec)


# Wrap the environment with DummyVecEnv
env = DummyVecEnv([lambda: env])

# Load the saved PPO model
model_path = "./ppo_carracing.zip"  # Path to your saved PPO model
model = PPO.load(model_path)

def to_csv(tuple_list):
    df = pd.DataFrame(data=tuple_list, columns=['Episode_num', 'Eval Score', 'Frames Used'])
    df.to_csv('./data/test_data_sb_model.csv', index = False)

def run_eval():
    eval_data = []

    for ep in range(1000):
        # Evaluate the model
        obs= env.reset()
        total_reward = 0
        for frame in range(1000):  # Run for 1000 steps (adjust as needed)
            action, _states = model.predict(obs, deterministic=True)  # Predict actions
            
            # print(env.step(action))
            obs, reward, done, info = env.step(action)  # Step through the environment
            total_reward += reward
            env.render()  # Render the environment (optional)
            if done:
                print(f'Ep {ep} Score: {round(total_reward[0], 2)} Frames: {frame}')
                break
        
        eval_data.append((ep, total_reward[0], frame))

    to_csv(eval_data)
    # Close the environment
    env.close()

run_eval()