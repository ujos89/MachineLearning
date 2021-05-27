"""
References:
TAXI environment github sourcecode and description
https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

DQN:
https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
https://wwiiiii.tistory.com/entry/Deep-Q-Network

Git repos:
https://github.com/gandroz/rl-taxi/blob/main/pytorch/taxi_demo_pytorch.ipynb
https://github.com/seungeunrho/minimalRL/blob/master/dqn.py
"""

import random
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import ks_2samp
from IPython.display import Image

torch.manual_seed(42)  # we fix the random seed for the same wieght initialization

# environment : gtx1060 3gb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Do not change the following parameters
GAMMA=0.9
EPISODES = 15000
MEMORY_SIZE = 50000

# Optimize the following parameters:
EPSILON=1.0

### we intentionally changed the following parameters away from their optimized values.
### Hence, you MUST optimize the following parameters in your way
BATCH_SIZE = 32
LEARNING_RATE = 0.01
TARGET_UPDATE = 5

'''
### Classes and Functions
- Class
    - DQN: Model class. There are two identical DQNs exist in this framework. You can change the shape of network.
         - current_dqn
         - target_dqn
    - Memory: Memory class. Implementing Experience Replay (ER) technique for DQN training
        - Save data gathered from current_dqn (called memory)
        - Give training data (randomly sampled from whole memory) to DQN
- Function
    - ε-greedy
    - Training
'''

env = gym.make("Taxi-v3")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=6)
        
        # He initialization for the weights (with ReLU)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = x.view(-1, 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

class Memory():
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        
    def put(self, transition):
        self.memory.append(transition)
        
    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        state_list, action_list, reward_list, nextState_list, done_list = [], [], [], [], []
        
        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            nextState_list.append(next_state)
            done_list.append([done])
            
        return torch.tensor(state_list, dtype=torch.float, device = device), \
                torch.tensor(action_list, device = device), \
                torch.tensor(reward_list, device = device), \
                torch.tensor(nextState_list, dtype=torch.float, device=device), \
                torch.tensor(done_list, device = device)
    
    def size(self):
        return len(self.memory)
    
    
def epsilon_greedy(q_function, epsilon):
    if random.random() > epsilon: # greedy
        return np.argmax(q_function.detach().numpy())
    else:
        return random.randint(0, 5)

def training(current_dqn, target_dqn, replay_memory, optimizer, gamma, batch_size):
    state, action ,reward, next_state, done = replay_memory.sample(batch_size)

    # Q(s_t, a)
    state_action_values = current_dqn(state).gather(1, action)
    # if episode terminates at next step, expected_state_action_values become reward
    next_state_values = target_dqn(next_state).max(1)[0].unsqueeze(1) * (~done).int()
    expected_state_action_values = reward + (GAMMA * next_state_values)

    # Huber loss compute
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    #model optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

current_dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(current_dqn.state_dict())
replay_memory = Memory()
optimizer_adam = optim.Adam(current_dqn.parameters(), lr=LEARNING_RATE)
epi_rewards, weight_diff = np.array([]), np.array([])
progress = 0

# for epi in tqdm(range(EPISODES)):
for epi in range(EPISODES):
    obs = env.reset()
    obs = [i for i in env.decode(obs)]
    done, total_reward = False, 0
    
    ## exploration strategies
    # constant epsilon greedy
    # eps = EPSILON
    # exponentially decaying epsilon greedy 
    eps = np.exp(-0.0005 * epi)

    while not done:
        state = np.array(obs)
        action = epsilon_greedy(current_dqn.forward(torch.from_numpy(state).float().to(device)), eps)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array([i for i in env.decode(next_state)])

        replay_memory.put([state, action, reward, next_state, done])
        state = next_state
        total_reward += reward

        # minimum standard to train
        if replay_memory.size() > BATCH_SIZE:
            training(current_dqn, target_dqn, replay_memory, optimizer_adam, GAMMA, BATCH_SIZE)

        # in gym, if number of state more than 200, terminate action to step 
        if done:
            break


    for name, param in current_dqn.named_parameters():
        x = torch.flatten(current_dqn.state_dict()[name]).to(device)
        y = torch.flatten(target_dqn.state_dict()[name]).to(device)
        loss = nn.MSELoss()
        mse = loss(x, y).cpu().numpy()
        weight_diff = np.append(weight_diff, mse)

    if epi % TARGET_UPDATE == 0 and epi != 0:
        target_dqn.load_state_dict(current_dqn.state_dict())

    epi_rewards = np.append(epi_rewards, total_reward)

    # display progress
    if int(epi/EPISODES * 100) >= progress + 1:
        print('progress: ', progress+1, '%')
        progress = int(epi/EPISODES*100)


print("Check whether model converges within 15000 episodes (in our codes, it successfully converges in 5000 episodes)")
print("    - Observe the weight changes to make sure that it converges (weight difference decreases).")
print("    - See the plot and check how fast does it converges.")

## graph visualization
# weight_diff (current_dqn vs. target_dqn)
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(weight_diff, label='weight difference (MSE)')
ax.legend(fontsize=20)
plt.savefig('Weight Difference(MSE).png')

# variance of rewards per epi
var = np.array([])
# window size: # of size for compute variance
wsize = 100
for i in range(len(epi_rewards) - wsize):
    var = np.append(var, np.var(epi_rewards[i:i + wsize]) / wsize)   
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(var, label='variance, wsize={}'.format(wsize))
ax.legend(fontsize=20)
plt.savefig('Variance of rewards.png'.format(wsize))

# rewards per epi
torch.save(current_dqn.state_dict(), '001-20_model.pth')
plt.figure(figsize=(20,4))
plt.plot(epi_rewards)
plt.savefig('epi_rewards.png')


# We are going to examine... 
# average reward after convergence (larger is better)
# variance after convergence (smaller is better)
# the last episode below the threshold (small is better)

# Below is an example of evaluation
## X: after conversion
X = -10000
threshold = -200  

avg_reward = np.mean(epi_rewards[X:-1])
print("Average reward (after convergence) is ", avg_reward)
var_reward = np.var(epi_rewards[X:-1])
print("Variance (after convergence) is ", var_reward)

for i in range(len(epi_rewards)):
    if(epi_rewards[i] <= threshold):
        max_epi_threshold = i
print("The last episode less than", threshold, "is", max_epi_threshold)