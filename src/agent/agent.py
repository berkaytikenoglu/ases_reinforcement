
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np

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

    def act(self, state):
        action_mean = self.actor(state)
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
        
        self.device = torch.device('cpu') # force cpu for simplicity on windows unless cuda checked
        self.policy.to(self.device)
        self.policy_old.to(self.device)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
            
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()
    
    def update(self, memory):
        # Convert list to tensor
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

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
