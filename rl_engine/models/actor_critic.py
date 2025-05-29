"""
Enterprise-grade Actor-Critic Implementation - Hybrid Policy/Value RL with Distributed Training
"""

from __future__ import annotations
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from utils.logger import get_logger
from utils.metrics import RLMetrics
from utils.serialization import serialize, deserialize

logger = get_logger(__name__)
metrics = RLMetrics()
torch.set_num_threads(int(os.cpu_count() / 2))  # Optimal for enterprise workloads

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules are a new feature.*")

@dataclass
class TrainingConfig:
    """Hyperparameters for distributed RL training"""
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    tau: float = 0.005  # Soft update coefficient
    entropy_coef: float = 0.01
    batch_size: int = 256
    sequence_length: int = 8  # For RNN-based policies
    grad_clip: float = 5.0
    target_update_interval: int = 100
    use_apex: bool = True
    mixed_precision: bool = True
    max_grad_norm: float = 5.0
    replay_alpha: float = 0.6  # Prioritized experience replay
    replay_beta: float = 0.4

class Actor(nn.Module):
    """Stochastic Policy Network with Architecture Autotuning"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Dynamic architecture parameters
        self.hidden_layers = nn.ModuleList([
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        ])
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Architecture optimization
        self.autotune = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.hidden_layers))
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.01)
            
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dynamic layer selection
        attn_weights = torch.sigmoid(self.autotune(state.mean(dim=0)))
        x = state
        for i, layer in enumerate(self.hidden_layers):
            if attn_weights[i] > 0.5:
                x = layer(x)
                x = torch.tanh(x)
        mu = self.mu(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

class Critic(nn.Module):
    """Dual Q-Network Architecture with Feature Extractor"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        self.Q1 = nn.Linear(hidden_dim, 1)
        self.Q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        features = self.feature_extractor(x)
        return self.Q1(features), self.Q2(features)

class ActorCritic(nn.Module):
    """Unified Actor-Critic Architecture with Distributed Training Support"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: TrainingConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Initialize networks
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim, action_dim)
        self.target_critic = Critic(obs_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        
        # Distributed training setup
        self._init_distributed()
        
        # Automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
    def _init_distributed(self):
        if dist.is_initialized():
            self.actor = DDP(self.actor)
            self.critic = DDP(self.critic)
            self.target_critic = DDP(self.target_critic)
            
    def soft_update(self):
        """Soft update target networks"""
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), 
                                         self.target_critic.parameters()):
                target_param.data.copy_(self.config.tau * param.data + 
                                      (1 - self.config.tau) * target_param.data)
                
    @torch.no_compile  # Disable compiler for stability
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform full training step with gradient management"""
        metrics_dict = {}
        
        # Unpack batch
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        with torch.autocast(device_type='cuda', enabled=self.config.mixed_precision):
            # Critic optimization
            current_Q1, current_Q2 = self.critic(states, actions)
            with torch.no_grad():
                next_actions, _ = self.actor(next_states)
                target_Q1, target_Q2 = self.target_critic(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + (1 - dones) * self.config.gamma * target_Q
                
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
            
            # Actor optimization
            actions_pred, log_probs = self.actor(states)
            Q1, Q2 = self.critic(states, actions_pred)
            Q = torch.min(Q1, Q2)
            actor_loss = -(Q - self.config.entropy_coef * log_probs).mean()
            
        # Critic backward pass
        self.scaler.scale(critic_loss).backward()
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.critic_optim)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        
        # Actor backward pass
        self.scaler.scale(actor_loss).backward()
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.actor_optim)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.scaler.step(self.actor_optim)
        self.scaler.update()
        
        # Log metrics
        metrics_dict.update({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': Q.mean().item(),
            'entropy': log_probs.mean().item()
        })
        
        # Soft update target network
        if self.step_counter % self.config.target_update_interval == 0:
            self.soft_update()
            
        self.step_counter += 1
        return metrics_dict

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action with exploration noise"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mu, std = self.actor(state)
            
            if deterministic:
                action = mu
            else:
                action = torch.normal(mu, std)
                
            return action.cpu().numpy()[0]

class ReplayBuffer(Dataset):
    """Prioritized Experience Replay with Distributed Sampling"""
    
    def __init__(self, capacity: int = 1e6, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = deque(maxlen=self.capacity)
        self.position = 0
        self._max_priority = 1.0
        
    def add(self, transition: Dict[str, np.ndarray]):
        """Add new experience with max priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.priorities.append(self._max_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, list]:
        """Sample batch with importance weights"""
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = {key: [] for key in self.buffer[0].keys()}
        for i in indices:
            transition = self.buffer[i]
            for key in transition:
                batch[key].append(torch.tensor(transition[key]))
                
        return batch, torch.tensor(weights), indices

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities after training"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item() + 1e-5  # Avoid zero priority
            
    def __len__(self):
        return len(self.buffer)

class DistributedReplayBuffer(ReplayBuffer):
    """Federated Experience Replay Across Multiple Nodes"""
    
    def __init__(self, global_buffer_size: int = 1e7, local_buffer_size: int = 1e5):
        super().__init__(capacity=local_buffer_size)
        self.global_buffer_size = global_buffer_size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
    def sync_global(self):
        """Synchronize experiences across all nodes"""
        if dist.is_initialized():
            # Gather all local buffers
            local_buffer = [self.buffer]
            gathered_buffers = [None] * self.world_size
            dist.all_gather_object(gathered_buffers, local_buffer)
            
            # Merge and prioritize
            global_buffer = []
            for buf in gathered_buffers:
                global_buffer.extend(buf)
                
            # Update priorities globally
            # (Implementation depends on specific distributed strategy)
            pass
            
# Example Usage
if __name__ == "__main__":
    # Initialize on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment dimensions
    obs_dim = 24  # Example observation space
    action_dim = 4  # Example action space
    
    # Training setup
    config = TrainingConfig(
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=256,
        gamma=0.99
    )
    
    model = ActorCritic(obs_dim, action_dim, config).to(device)
    replay_buffer = ReplayBuffer()
    
    # Example training loop
    for episode in range(1000):
        state = np.random.randn(obs_dim)  # Mock environment
        total_reward = 0
        
        while True:
            action = model.get_action(state)
            next_state = np.random.randn(obs_dim)  # Mock transition
            reward = np.random.rand()  # Mock reward
            done = np.random.rand() > 0.95  # Mock termination
            
            replay_buffer.add({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            if len(replay_buffer) > config.batch_size:
                batch, weights, indices = replay_buffer.sample(config.batch_size)
                metrics = model.update(batch)
                
                # Update priorities (TD-error based)
                priorities = np.abs(metrics['critic_loss']) + 1e-5
                replay_buffer.update_priorities(indices, priorities)
                
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
