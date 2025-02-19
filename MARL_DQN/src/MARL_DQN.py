import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from log_DQN import TensorboardLogger

class DQN(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)  # Outputs Q-values for all actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Args:
        capacity (int): Maximum size of buffer
    """
    def __init__(self, capacity, state_dim, action_dim, n_agents):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
    
    
class MultiAgentDQN:
    """
    Multi-Agent Deep Q-Network with shared replay buffer and centralized training.

    Args:
        n_agents (int): Number of agents
        state_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        lr (float): Learning rate
        gamma (float): Discount factor
    """
    def __init__(self, buffer, n_agents, state_dim, action_dim, lr=1e-3, t_target_interval=5000, gamma=0.99):
        self.buffer = buffer
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.t_target_interval = t_target_interval
        self.update_step = 1
        self.gamma = gamma

        # Initialize Q-network and target network
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Loss Logger
        self.logger = TensorboardLogger(n_agents=n_agents)

    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Observation of the environment
            epsilon: Exploration rate

        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
   
        if random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            return action

    def update(self, batch_size): 
        """
        Update Q-network using sampled transitions from replay buffer.

        Args:
            batch_size (int): Number of samples to update from
        """
        if len(self.buffer) < batch_size:
            return
        else:
            # Sample batch
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            
            # Convert to tensors
            states = states
            actions = actions.unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
            next_states = next_states
            dones = dones.unsqueeze(-1)

            # Q-values for current states
            q_values = self.q_network(states).gather(2, actions)  # self.q_network(states)는 (3, action_dim) 크기의 Q-값 행렬을 반환 / .gather(1, actions)를 통해, 각 에이전트가 선택한 행동에 해당하는 Q-값만 가져옴  
            # Target Q-values for next states
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(dim=2)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                target_q_values = target_q_values.unsqueeze(-1)

            # Compute loss and update network
            #loss = nn.MSELoss()(q_values, target_q_values)
            loss = nn.SmoothL1Loss()(q_values, target_q_values) 
            self.optimizer.zero_grad()
            loss.backward()  # pytorch에서는 loss값이 여러 개여도 각 loss값들을 합산하여 평균내서 한 번의 업데이트를 진행함.   
            self.optimizer.step()
            
            # Increment update step
            self.update_step += 1
            
            # Loss Logger
            self.logger.log_loss(loss.item(), self.update_step)

            # Update target network periodically
            if self.update_step % self.t_target_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
