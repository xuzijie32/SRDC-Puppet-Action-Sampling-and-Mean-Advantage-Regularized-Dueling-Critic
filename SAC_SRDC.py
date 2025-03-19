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
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.max_action = max_action

        self.except_times = 10

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        mu = self.fc_mu(x2)
        std = F.softplus(self.fc_std(x2))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.max_action
        return action, log_prob.sum(1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Critic, self).__init__()

        # Q1 architecture
        self.s1 = nn.Linear(state_dim , 256)
        self.v2 = nn.Linear(256, 256)
        self.v3 = nn.Linear(256, 1)
        self.a1 = nn.Linear( 256 + action_dim, 256)
        self.a2 = nn.Linear(256, 256)
        self.a3 = nn.Linear(256, 1)
        self.max_action = max_action

    def forward(self, state, action, num_sample):
        s = F.relu(self.s1(state))
        v = F.relu(self.v2(s))
        v = self.v3(v)
        sa = torch.cat([s, action], 1)
        a = F.relu(self.a1(sa))
        a = F.relu(self.a2(a))
        a = self.a3(a)

        mean = torch.zeros_like(a)
        if num_sample != 0:
            for i in range(num_sample):
                noise = torch.rand_like(action).to(device)
                noise_ = self.max_action * (noise * 2 - 1)
                sa1 = torch.cat([s, noise_], 1)
                a_1 = F.relu(self.a1(sa1))
                a_1 = F.relu(self.a2(a_1))
                a_1 = self.a3(a_1)
                mean += a_1 / num_sample

        return v + a, mean

    def Q1(self, state, action):
        s = F.relu(self.s1(state))
        v = F.relu(self.v2(s))
        v = self.v3(v)
        sa = torch.cat([s, action], 1)
        a = F.relu(self.a1(sa))
        a = F.relu(self.a2(a))
        a = self.a3(a)

        return v,a


class SAC_SRDC(object):

    def __init__(self, state_dim, action_dim, max_action,target_entropy,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, tau=0.001, discount=0.99, update_interval=1, num_sample=5, adaptive=True,alpha_SRDC=0.75):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim,max_action).to(device)
        self.critic_2 = Critic(state_dim, action_dim,max_action).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim,max_action).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim,max_action).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.update_interval = update_interval
        self.target_entropy = target_entropy
        self.discount = discount
        self.tau = tau
        self.num_sample=num_sample
        self.adaptive=adaptive
        self.alpha = alpha_SRDC

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_, _ = self.actor(state)
        return action_.cpu().data.numpy().flatten()

    def train(self, replaybuffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replaybuffer.sample(batch_size)

        with torch.no_grad():
            next_actions, log_prob = self.actor(next_state)
            entropy = -log_prob
            v1, a1 = self.target_critic_1.Q1(next_state, next_actions)
            v2, a2 = self.target_critic_2.Q1(next_state, next_actions)
            q1_value = v1 + a1
            q2_value = v2 + a2
            next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
            target_Q = reward + not_done * self.discount * next_value

        current_Q1 , current_mean1 = self.critic_1(state, action, self.num_sample)
        current_Q2 , current_mean2 = self.critic_2(state, action, self.num_sample)
        baseline = torch.zeros_like(current_mean1)

        critic_1_Q_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_Q_loss = F.mse_loss(current_Q2, target_Q)
        critic_1_R_loss = F.mse_loss(current_mean1, baseline)
        critic_2_R_loss = F.mse_loss(current_mean2, baseline)
        if self.adaptive:
            with torch.no_grad():
                alpha = (critic_1_Q_loss + critic_2_Q_loss) / (
                            critic_1_Q_loss + critic_2_Q_loss + critic_1_R_loss + critic_2_R_loss)
                beta = 1 - alpha
            critic_1_loss = alpha * critic_1_Q_loss + beta * critic_1_R_loss
            critic_2_loss = alpha * critic_2_Q_loss + beta * critic_2_R_loss
        else:
            critic_1_loss = self.alpha * critic_1_Q_loss + (1-self.alpha) * critic_1_R_loss
            critic_2_loss = self.alpha * critic_2_Q_loss + (1-self.alpha) * critic_2_R_loss
        critic_Q_loss = critic_1_Q_loss + critic_2_Q_loss
        critic_R_loss = critic_1_R_loss + critic_2_R_loss
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()

        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        v1 , a1 = self.critic_1.Q1(state, new_actions)
        v2 , a2 = self.critic_2.Q1(state, new_actions)
        q1_value = v1 + a1
        q2_value = v2 + a2
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()
        actor_v = torch.mean(0.5 * (v1 + v2))
        actor_a = torch.mean(0.5 * (a1 + a2))
        actor_q = torch.mean(torch.min(q1_value, q2_value))

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()

        self.log_alpha_optimizer.step()

        # Update the frozen target models
        if self.total_it % self.update_interval == 0:
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return float(critic_Q_loss), float(critic_R_loss), float(actor_v), float(actor_a), float(actor_q), float(-actor_loss), float(alpha_loss)

    def save(self, filename):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer")

        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_2_optimizer.state_dict(), filename + "_critic_2_optimizer")

        torch.save(self.target_critic_1.state_dict(), filename + "_target_critic_1")
        torch.save(self.target_critic_2.state_dict(), filename + "_target_critic_2")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.log_alpha, filename + "_alpha")
        torch.save(self.log_alpha_optimizer.state_dict(), filename + "_alpha_optimizer")