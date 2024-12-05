import numpy as np
import gymnasium as gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pandas as pd
import math

torch.manual_seed(0)
np.random.seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)


# --------------- HYPER PARAMETERS ---------------#
# ------------------------------------------------#

img_stack_len = 3               # number of frames stacked
repeat_actions = 8              # number of times to repeat an action
gamma = 0.99                    # discounting
print_steps = 10                # number of episodes to skip before printing

# ------------------------------------------------#


# ----------- Utils classes/functions ------------#
# ------------------------------------------------#
class Memory():
    def __init__(self):
        self.count = 0
        self.length = 100
        self.memory_arr = np.zeros(self.length)

    def update_and_return_avg(self, reward):
        self.memory_arr[self.count] = reward
        self.count = (self.count+1) % self.length

        return np.mean(self.memory_arr)


def to_csv(tuple_list):
    df = pd.DataFrame(data=tuple_list, columns=['Episode_num', 'Score', 'Moving average'])
    df.to_csv('./data/train_data.csv', index=False)
# ------------------------------------------------#



''' Define custom data type for a transition. Consists of
## state (4-stack grayscaled images)
## action (3-tuple of (steering, gas, brake))
## log probability of action (scalar)
## reward (scalar)
## next state (4-stack grayscaled images) '''
transition = np.dtype([('s', np.float64, 
                       (img_stack_len, 96, 96)), 
                       ('a', np.float64, (3,)), 
                       ('a_logp', np.float64),
                       ('r', np.float64), 
                       ('s_', np.float64, (img_stack_len, 96, 96))])

'''CNN for Actor-Critic PPO'''
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.common_cnn = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack_len, 8, kernel_size=4, stride=2),
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

'''Custom wrapper around gym environment. Does the following :
## Helps with 4-stacking grayscaled images'''
class WrapperEnv(gym.Env):

    def __init__(self, stable_baseline = False):
        
        self.stable_basline = stable_baseline
        self.img_dtype = np.float32 if not self.stable_basline else np.uint8
        
        self.env = gym.make('CarRacing-v3')
        self.reward_threshold = self.env.spec.reward_threshold - 100

        #NOTE: Delete later if need be 

        self.action_space = self.env.action_space  # Continuous [-1, 1] for steering, throttle, brake

        if not self.stable_basline:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(img_stack_len, 96, 96), dtype=self.img_dtype
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(96, 96, img_stack_len), dtype=self.img_dtype
            )

    def reset(self):
        self.counter = 0

        # check if agent recently receives reward and terminate otherwise
        self.recent_avg_reward = Memory()

        # set die to False
        self.die = False
        
        # greyscale the given image
        rgb_img, _ = self.env.reset()
        grayscaled_img = self.grayscale(rgb_img)
        self.img_stack = [grayscaled_img] * img_stack_len  # four frames for decision

        # return initial_state, info
        state = np.stack(self.img_stack, axis=0 if not self.stable_baseline else -1)
        return state, {}

    def step(self, action):
        total_reward = 0
        for _ in range(repeat_actions):
            rgb_img, reward, done, trunc, _ = self.env.step(action)

            # green penalty
            if np.mean(rgb_img[:, :, 1]) > 185.0: reward -= 0.05

            # don't penalize dying
            if done: reward += 100

            total_reward += reward

            # if no reward recently, end the episode
            no_reward = True if self.recent_avg_reward(reward) <= -0.5 else False

            if no_reward or done:
                break

        # get new state    
        grayscaled_img = self.grayscale(rgb_img)
        assert grayscaled_img.shape == (96, 96), f"Image shape is incorrect: {grayscaled_img.shape}"

        # append new state 
        self.img_stack.pop(0)
        self.img_stack.append(grayscaled_img)
        assert len(self.img_stack) == img_stack_len, f"stack length is : {len(self.stack)}"
        state = np.stack(self.img_stack, axis=0 if not self.stable_basline else -1)
        return state, total_reward, done, no_reward, {}

    def grayscale(self, rgb):
        grayscaled = np.dot(rgb[..., :], [0.299, 0.587, 0.114]) # grayscale it
        # normalize it
        if self.stable_basline: grayscaled = grayscaled / 128. - 1.
        return grayscaled

    def render(self, *arg):
        self.env.render(*arg)
   
    def close(self):
        self.env.close()

'''Car Agent to train the CNN'''
class CarAgent():
    
    def __init__(self):
        self.training_step = 0
        self.net = CNN().double().to(device)

        self.buffer_len = 2000
        self.batch_size = 128

        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.clip_val = 0.1
        self.num_iters = 10

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    # select action based on state
    def select_action(self, state):
        # reshape state to be batch-format like
        state = torch.from_numpy(state).double().unsqueeze(0)

        # get parameters
        with torch.no_grad():
            alpha, beta = self.net(state)[0]

        # select action with parameters
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        # turn tensor to np array and return
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    # update network
    def update(self):
        print('updating')
        self.training_step += 1

        # get (s, a, r, s_, a_logp) from buffer
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        # calculate advantage
        with torch.no_grad():
            # critic/target value
            next_state_values = self.net(s_)[1]
            target_val = r + gamma * next_state_values

            # adv = target - (actor value)
            adv = target_val - next_state_values

        # run ppo #epoch times
        for _ in range(self.num_iters):

            for batch in np.array_split(np.random.permutation(np.arange(self.buffer_capacity)), math.ceil(self.buffer_capacity / self.batch_size)):
                
                # get the ratio of new/old prob for action
                ab, value = self.net(s[batch])
                alpha, beta = ab
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[batch]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[batch])

                target_loss = F.smooth_l1_loss(value, target_val[batch])
                loss = ratio * adv[batch]
                clipped_loss = torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val) * adv[batch]
                actor_loss = -torch.min(loss, clipped_loss).mean()
                loss = actor_loss + 2. * target_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # store transition in replay buffer
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False


    # save model params
    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')


# code to start training the agent
def train_agent():
    agent = CarAgent()
    env = WrapperEnv()

    moving_avg = 0
    state = env.reset()

    # will contain a 4-tuple of (ep, score, running_score, loss)
    tuple_list = []
    score_counter = 0

    for ep in range(10000):
        score = 0
        state = env.reset()

        for _ in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, no_reward, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

            if agent.store((state, action, a_logp, reward, state_)): agent.update()

            score += reward
            state = state_
            if done or no_reward:
                break
        moving_avg = moving_avg * 0.99 + score * 0.01

        data_tuple = (ep, score, moving_avg)
        tuple_list.append(data_tuple)
        
        if score > env.reward_threshold : score_counter+=1

        if ep % print_steps == 0:
            agent.save_param()
            print(f'Episode: {ep}   Score: {round(score, 2)}    Moving Average: {round(moving_avg, 2)}')
            f = open('/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/data/train.txt', 'a')
            f.write('\n'.join(str(t) for t in tuple_list))
            f.write('\n')
            tuple_list = []

        if score_counter >= 5:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            f.write('\n'.join(str(t) for t in tuple_list))
            f.write('\n')
            to_csv(tuple_list)
            
            break

    to_csv(tuple_list)


if __name__ == "__main__":
    train_agent()