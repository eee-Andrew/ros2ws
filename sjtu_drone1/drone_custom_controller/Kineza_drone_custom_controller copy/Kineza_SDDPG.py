# sddpg_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Class to write the actor and critic networks
# Actor network takes state information and outputs actions
# Critic network takes state information and actions, outputs Q-values
# Both networks use ReLU activations and Xavier initialization for weights
# The actor network has a Tanh output layer to ensure actions are in the range [-1, 1]
"""
The internal state (Su = [θt,vt,ωt]) has 3 dimensions, 1)direction relative to north, 2) velocity, 3) angular velocity
So =12 lidar readings
sg = 4 goal state , 2 is the relative distance to the goal, 2 is the relative angle to the goal (horizontal and vertical)
""" 
class Actor(nn.Module):
    def __init__(self, si_dim=3, so_dim=12, sg_dim=3, action_dim=3):  # si_dim is the state_independent dimension
                                                                      # so_dim is the state observation dimension
                                                                        # sg_dim is the state goal dimension
        super(Actor, self).__init__()
        self.obs_branch = nn.Sequential(
            nn.Linear(so_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU())
        self.goal_branch = nn.Sequential(
            nn.Linear(sg_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU())
        self.global_net = nn.Sequential(
            nn.Linear(16 + 16 + si_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh())
        self.apply(weights_init_)
# In this function we initialize the def_forward fucntion.
#This function is used to 
    def forward(self, si, so, sg):
        ao = self.obs_branch(so)
        ag = self.goal_branch(sg)
        x = torch.cat([ao, ag, si], dim=-1)
        return self.global_net(x)       

class Critic(nn.Module):
    def __init__(self, si_dim=3, so_dim=12, sg_dim=3, action_dim=3):
        super(Critic, self).__init__()
        input_dim = si_dim + so_dim + sg_dim + action_dim
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.apply(weights_init_)

    def forward(self, si, so, sg, action):
        x = torch.cat([si, so, sg, action], dim=-1)
        return self.q_net(x)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class SDDPG:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        
        # State splitting dimensions - adjust based on actual observation
        so_dim, sg_dim, si_dim = 12, 3, 3  # <-- fixed si_dim
        
        self.actor = Actor(si_dim, so_dim, sg_dim, action_dim).to(device)
        self.actor_target = Actor(si_dim, so_dim, sg_dim, action_dim).to(device)
        self.critic = Critic(si_dim, so_dim, sg_dim, action_dim).to(device)
        self.critic_target = Critic(si_dim, so_dim, sg_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=100000)
        # Initialize noise for exploration
        self.noise = OUNoise(action_dim)
        self.gamma = 0.99
        self.tau = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.7
        self.epsilon_min = 0.05

    def select_action(self, state, explore=True):
        # Debug: Print state information
        print(f"\n=== DEBUG STATE INFO ===")
        print(f"Input state shape: {np.array(state).shape}")
        print(f"Input state: {state}")
        
        state = torch.FloatTensor(state).to(self.device)
        print(f"Tensor state shape: {state.shape}")
        
        # Check if state has expected dimensions
        if len(state) < 18:
            print(f"ERROR: State has only {len(state)} dimensions, expected 18")
            return np.array([0.0, 0.0, 0.0])  # Return zero action as fallback
        
        # State splitting based on environment's get_obs() method:
        # return np.concatenate([self.lidar_data, self.current_position, delta]).astype(np.float32)
        so = state[:12].unsqueeze(0)        # LiDAR data (12 values)
        si = state[12:15].unsqueeze(0)      # Current position (3 values)  
        sg = state[15:18].unsqueeze(0)      # Delta to goal (3 values)
        
        print(f"so (LiDAR) shape: {so.shape}, values: {so}")
        print(f"si (position) shape: {si.shape}, values: {si}")
        print(f"sg (goal delta) shape: {sg.shape}, values: {sg}")
        print(f"=========================\n")
        
        # Check for empty tensors
        if so.shape[1] == 0 or si.shape[1] == 0 or sg.shape[1] == 0:
            print("ERROR: One of the state components is empty!")
            return np.array([0.0, 0.0, 0.0])
        
        try:
            action = self.actor(si, so, sg).cpu().data.numpy().flatten()
            if explore:
                action += self.epsilon * self.noise.sample()
            return np.clip(action, -1.0, 1.0)
        except Exception as e:
            print(f"ERROR in actor forward pass: {e}")
            return np.array([0.0, 0.0, 0.0])

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        s, a, r, s_, d = self.replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(1 - d).unsqueeze(1).to(self.device)

        # State splitting for batch processing
        so, si, sg = s[:, :12], s[:, 12:15], s[:, 15:18]
        so_, si_, sg_ = s_[:, :12], s_[:, 12:15], s_[:, 15:18]

        with torch.no_grad():
            a_ = self.actor_target(si_, so_, sg_)
            q_target = self.critic_target(si_, so_, sg_, a_)
            y = r + d * self.gamma * q_target

        q = self.critic(si, so, sg, a)
        critic_loss = F.smooth_l1_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_pred = self.actor(si, so, sg)
        actor_loss = -self.critic(si, so, sg, a_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Decay exploration noise
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)