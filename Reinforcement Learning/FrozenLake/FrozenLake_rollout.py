import gym
import numpy as np
import random

from pprint import pprint
from tqdm import tqdm_notebook as tqdm  # you may need to install tqdm by "pip install tqdm"
from itertools import cycle, count
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

##Functions 

# These codes are developed by Miguel Morales
# Visit: https://github.com/mimoralea/gdrl

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
    return

#change print_state_value_function for convenience of visualize
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(11), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(8), end=" ")
        if (s + 1) % n_cols == 0: print("|")
    return

def print_action_value_function(Q, 
                                optimal_Q=None, 
                                action_symbols=('<', '>'), 
                                prec=3, 
                                title='Action-value function:'):
    vf_types=('',) if optimal_Q is None else ('', '*', 'err')
    headers = ['s',] + [' '.join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q-Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))
    return

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)


##Import FrozenLake-v0

env = gym.make('FrozenLake-v0')

#P = env.env.P
init_state = env.reset()
goal_state = 15

LEFT, DOWN, RIGHT, UP = range(4)

# This code generates the MDP kernel 
# we consider a deterministic environment (i.e., transition probability = 1.0)
# action(0(L), 1(D), 2(R), 3(U)): [(prob, next state, reward, done)]

detP = {}
for i in range(16):
    a = {}
    # 0
    if i % 4 == 0: # left wall
        a[0] = [(1, -1, -100.0, True)]
    else:
        # check penalty or terminal state
        if i - 1 in [5, 7]:
            a[0] = [(1, i - 1, -10.0, False)]
        elif i - 1 in [10, 12]:
            a[0] = [(1, i - 1, -5.0, False)]
        elif i - 1 == 15:
            a[0] = [(1, i - 1, 1.0, True)]
        # plain state
        else:
            a[0] = [(1, i - 1, 0.0, False)]
            
    # 1
    if i + 4 > 15: # bottom wall
        a[1] = [(1, -1, -100.0, True)]
    else:
        if i + 4 in [5, 7]:
            a[1] = [(1, i + 4, -10.0, False)]
        elif i + 4 in [10, 12]:
            a[1] = [(1, i + 4, -5.0, False)]
        elif i + 4 == 15:
            a[1] = [(1, i + 4, 1.0, True)]
        else:
            a[1] = [(1, i + 4, 0.0, False)]
            
    # 2
    if i % 4 == 3: # right wall
        a[2] = [(1, -1, -100.0, True)]
    else:
        if i + 1 in [5, 7]:
            a[2] = [(1, i + 1, -10.0, False)]
        elif i + 1 in [10, 12]:
            a[2] = [(1, i + 1, -5.0, False)]
        elif i + 1 == 15:
            a[2] = [(1, i + 1, 1.0, True)]
        else:
            a[2] = [(1, i + 1, 0.0, False)]
            
    # 3
    if i - 4 < 0: # upper wall
        a[3] = [(1, -1, -100.0, True)]
    else:
        if i - 4 in [5, 7]:
            a[3] = [(1, i - 4, -10.0, False)]
        elif i - 4 in [10, 12]:
            a[3] = [(1, i - 4, -5.0, False)]
        elif i - 4 == 15:
            a[3] = [(1, i - 4, 1.0, True)]
        else:
            a[3] = [(1, i - 4, 0.0, False)]

    if i == 15:
        a = {
            0: [(1.0, 15, 0, True)],
            1: [(1.0, 15, 0, True)],
            2: [(1.0, 15, 0, True)],
            3: [(1.0, 15, 0, True)]}
        
    detP[i] = a
    
P = detP
env.env.P = P
#P

def print_R (P):
    sv_up, sv_middle, sv_down = "","",""
    print("------------------------------------------------------------------")
    for s in range(16):
        _,_,v_up,_ = P[s][UP][0]
        _,_,v_left,_ = P[s][LEFT][0] 
        _,_,v_right,_ = P[s][RIGHT][0]
        _,_,v_down,_ = P[s][DOWN][0]
        sv_up = sv_up + str(v_up).center(16, " ")
        sv_middle = sv_middle + str(v_left).center(8, " ") + str(v_right).center(8, " ")
        sv_down = sv_down + str(v_down).center(16," ")
        if( (s+1) % 4 == 0):
            print(sv_up)
            print(sv_middle)
            print(sv_down)
            print("------------------------------------------------------------------")
            sv_up, sv_middle, sv_down = "","",""

print("Reward for each (s,a)")
print_R(P)
print()

## Base PI

LEFT, DOWN, RIGHT, UP = range(4)

# base policy
base_pi = lambda s: {
    0:RIGHT, 1:RIGHT, 2:RIGHT, 3:DOWN,
    4:DOWN, 5:DOWN, 6:RIGHT, 7:DOWN,
    8:DOWN, 9:DOWN, 10:DOWN, 11:DOWN,
    12:RIGHT, 13:RIGHT, 14:RIGHT, 15:LEFT
}[s]

print("<base_pi>")
print_policy(base_pi, P)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, base_pi, goal_state=goal_state)*100, 
    mean_return(env, base_pi)))
print()

##Policy Evaluation, policy improvement, policy iteration, value iteration

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s:pi(s) for s in range(len(P))}:
            break
    return V, pi

def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi

# example with go_get_pi
V = policy_evaluation(base_pi, P, gamma=0.99)
print_state_value_function(V, P, prec=4)
print()

## Rollout Algorithm
def do_rollout(current_pi,
                   base_pi,
                   env, 
                   current_state, 
                   gamma=1.0, 
                   max_steps=100):

    action_value = [0] * 4

    # a:direction {0:left, 1:down, 2:right, 3:up}
    for a in range(4):
        # initalize env
        now_state = env.reset()
        
        # move to current state
        while now_state != current_state:
            nxt_state, _, _, _ = env.step(current_pi(now_state))
            now_state = nxt_state

        # get action in current state
        nxt_state, reward, done, _ = env.step(a)
        action_value[a] = reward

        # if action make out of frozenlake, state become -1
        if nxt_state == -1:
            nxt_state = now_state
            # move to current state again for simulate env
            now_state = env.reset()
            while now_state != nxt_state:
                nxt_state, _, _, _ = env.step(current_pi(now_state))
                now_state = nxt_state
        else:
            now_state = nxt_state

        # lookahead with base pi
        for iters in range(max_steps):
            nxt_state, reward, done, _ = env.step(base_pi(now_state))
            now_state = nxt_state
            action_value[a] += gamma * reward
            
            if done:
                break
    
    # print("q:", action_value)
    # choose proposal action with rollout
    return action_value.index(max(action_value))
    
    
def online_rollout(env, base_pi, sim_env):
    nS = env.observation_space.n
    pi = {state:base_pi(state) for state in range(nS)}    

    ncount = 0
    reward = 0
    state, done = env.reset(), False

    while not done:
        current_pi = lambda s: {s:pi[s] for s in range(nS)}[s]
        pi[state] = do_rollout(current_pi, base_pi, sim_env, state, gamma=0.99, max_steps=100)
        next_state, reward, done, _ = env.step(pi[state])
        print("#", ncount, ": (s a s')=(", state, pi[state], next_state, "), r =", reward)
        ncount = ncount+1
        state = next_state
        
    return_pi = lambda s: {s:pi[s] for s in range(nS)}[s]
    return return_pi

sim_env = gym.make('FrozenLake-v0')  # Need to run the base policy over this simulation environment
sim_env.env.P = P
rollout_pi = online_rollout(env, base_pi, sim_env)

print()
print("Base policy")
print_policy(base_pi, P)
V = policy_evaluation(base_pi, P, gamma=0.99)
print_state_value_function(V, P, prec=4)

print()
print("Rollout policy")
print_policy(rollout_pi, P)
V = policy_evaluation(rollout_pi, P, gamma=0.99)
print_state_value_function(V, P, prec=4)