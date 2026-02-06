
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import Params

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim_continuous, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        self.action_std = action_std_init
        
        # Actor Network (Predicts actions)
        # Input: Observation
        # Output: Mean of the action distribution (for continuous action: Angle) 
        # AND a separate head for Fire Trigger (Discrete-ish, but we modeled it as continuous 0.0-1.0)
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim_continuous) # Outputs: mean of angle, mean of trigger
        )
        
        # Critic Network (Predicts Value of state)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.action_var = torch.full((action_dim_continuous,), action_std_init * action_std_init)
        
    def forward(self):
        raise NotImplementedError
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = torch.full((self.actor[-1].out_features,), new_action_std * new_action_std).to(self.actor[-1].weight.device)

    def act(self, state, deterministic=False):
        action_mean = self.actor(state)
        
        if deterministic:
            # Test Mode: Use the mean directly (No noise)
            action = action_mean
            action_logprob = torch.zeros(1).to(action.device) # Dummy logprob
        else:
            # Training Mode: Sample from distribution (Exploration)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(state.device)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(state.device)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class Agent:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).float()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # GPU Support - use CUDA if available (can be forced to CPU)
        self.device = torch.device('cpu')  # Default
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[GPU] CUDA ENABLED: {torch.cuda.get_device_name(0)}")
        else:
            print("[CPU] GPU not available, using CPU")
            
        self.policy.to(self.device)
        self.policy_old.to(self.device)

    def reset_feature_weights(self, indices):
        """Reset weights for specific input features to zero to prevent transfer shock"""
        for model in [self.policy, self.policy_old]:
            # Reset Actor input weights (Layer 0 is Linear)
            model.actor[0].weight.data[:, indices] = 0.0
            # Reset Critic input weights (Layer 0 is Linear)
            model.critic[0].weight.data[:, indices] = 0.0
        print(f"DEBUG: Reset weights for input features: {indices}")
        self.policy_old.to(self.device)
        self.policy.action_var = self.policy.action_var.to(self.device)
        self.policy_old.action_var = self.policy_old.action_var.to(self.device)
        
    def set_action_std(self, new_action_std):
        """Update exploration noise std for both current and old policies"""
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
    
    def set_device(self, use_gpu=True):
        """Switch between CPU and GPU"""
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[GPU] Switched to CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[CPU] Using CPU mode")
        
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.policy.action_var = self.policy.action_var.to(self.device)
        self.policy_old.action_var = self.policy_old.action_var.to(self.device)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state, deterministic=deterministic)
            
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of returns => REPLACED WITH GAE
        
        # Get old states
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # Get Values for GAE
        with torch.no_grad():
            values = self.policy.critic(old_states).squeeze()
            
        # GAE Calculation
        rewards = []
        gae = 0
        advantages = []
        
        # Reverse iteration
        for i in reversed(range(len(memory.rewards))):
            if memory.is_terminals[i]:
                next_val = 0
                gae = 0
            else:
                if i < len(memory.rewards) - 1:
                    next_val = values[i + 1]
                else:
                    next_val = values[i] # Approximation for last step
            
            delta = memory.rewards[i] + Params.GAMMA * next_val - values[i]
            gae = delta + Params.GAMMA * Params.LAMBDA * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Returns = Advantage + Value
        returns = advantages + values
        
        # Advantage Normalization
        if getattr(Params, 'ADVANTAGE_NORMALIZE', True):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Mini-batch processing
            batch_size = getattr(Params, 'MINIBATCH_SIZE', 64)
            data_size = old_states.size(0)
            
            # Generate random indices
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, data_size, batch_size):
                end_idx = min(start_idx + batch_size, data_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                mb_states = old_states[batch_indices]
                mb_actions = old_actions[batch_indices]
                mb_logprobs = old_logprobs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - mb_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_advantages

                # Final loss: Policy Loss + Value Loss + Entropy Loss
                loss = -torch.min(surr1, surr2) + \
                       Params.VALUE_COEF * self.MseLoss(state_values, mb_returns) - \
                       Params.ENTROPY_COEF * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def decay_lr(self, current_ep, max_ep):
        """Linear learning rate decay"""
        if getattr(Params, 'LR_SCHEDULE', 'linear') == 'linear':
            # Linear decay from Initial LR to 0 (or small value)
            frac = 1.0 - (current_ep / max_ep)
            lr = Params.LEARNING_RATE * frac
            lr = max(lr, 1e-7) # Min safety
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
