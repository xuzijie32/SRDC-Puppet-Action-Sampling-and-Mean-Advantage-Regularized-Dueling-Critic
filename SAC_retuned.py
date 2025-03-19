import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.max_action
        return action, log_prob.sum(1,keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256 + action_dim, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state, action):

        q1 = F.relu(self.l1(state))
        sa1 = torch.cat([q1, action], 1)
        q1 = F.relu(self.l2(sa1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        return q1

    def Q1(self, state, action):
        q1 = F.relu(self.l1(state))
        sa1 = torch.cat([q1, action], 1)
        q1 = F.relu(self.l2(sa1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

class SAC_retuned(object):

    def __init__(self, state_dim, action_dim, max_action,target_entropy,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, tau=0.001, discount=0.99, update_interval=1):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy = target_entropy
        self.update_interval=update_interval
        self.discount = discount
        self.tau = tau

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_ , _= self.actor(state)
        return action_.cpu().data.numpy().flatten()

    def train(self, replaybuffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replaybuffer.sample(batch_size)

        with torch.no_grad():
            next_actions, log_prob = self.actor(next_state)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_state, next_actions)
            q2_value = self.target_critic_2(next_state, next_actions)
            next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
            target_Q = reward + not_done * self.discount * next_value

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        q1_value = self.critic_1(state, new_actions)
        q2_value = self.critic_2(state, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.total_it % self.update_interval == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return float(critic_loss), float(-actor_loss)

    def save(self, filename):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer")

        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_2_optimizer.state_dict(), filename + "_critic_2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
