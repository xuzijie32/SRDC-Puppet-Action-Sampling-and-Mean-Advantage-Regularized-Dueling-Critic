import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,num_sample):
        super(Critic, self).__init__()

        # Q1 architecture
        self.s1 = nn.Linear(state_dim, 256)
        self.v2 = nn.Linear(256, 256)
        self.v3 = nn.Linear(256, 1)
        self.a1 = nn.Linear(action_dim + 256, 256)
        self.a2 = nn.Linear(256, 256)
        self.a3 = nn.Linear(256, 1)

        # Q2 architecture
        self.s4 = nn.Linear(state_dim, 256)
        self.v5 = nn.Linear(256, 256)
        self.v6 = nn.Linear(256, 1)
        self.a4 = nn.Linear(action_dim + 256, 256)
        self.a5 = nn.Linear(256, 256)
        self.a6 = nn.Linear(256, 1)

        self.num_sample = num_sample

    def forward(self, state, action, compute_mean, mu, std, max_action):
        s1 = F.relu(self.s1(state))
        s2 = F.relu(self.s4(state))

        sa1 = torch.cat([s1, action], 1)
        sa2 = torch.cat([s2, action], 1)

        v1 = F.relu(self.v2(s1))
        v1 = self.v3(v1)
        a1 = F.relu(self.a1(sa1))
        a1 = F.relu(self.a2(a1))
        a1 = self.a3(a1)

        v2 = F.relu(self.v5(s2))
        v2 = self.v6(v2)
        a2 = F.relu(self.a4(sa2))
        a2 = F.relu(self.a5(a2))
        a2 = self.a6(a2)

        mean1 = torch.zeros_like(a1)
        mean2 = torch.zeros_like(a2)
        if compute_mean:
            for i in range(self.num_sample):
                dist = Normal(mu, std * max_action)
                noise_ = (dist.rsample()).clip(-max_action, max_action)
                sa1 = torch.cat([s1, noise_], 1)
                sa2 = torch.cat([s2, noise_], 1)
                a_1 = F.relu(self.a1(sa1))
                a_1 = F.relu(self.a2(a_1))
                a_1 = self.a3(a_1)
                a_2 = F.relu(self.a4(sa2))
                a_2 = F.relu(self.a5(a_2))
                a_2 = self.a6(a_2)
                mean1 += a_1 / self.num_sample
                mean2 += a_2 / self.num_sample

        return a1 + v1 , a2 + v2 , mean1 , mean2

    def Q1(self, state, action):
        s = F.relu(self.s1(state))
        sa = torch.cat([s, action], 1)
        v1 = F.relu(self.v2(s))
        v1 = self.v3(v1)
        a1 = F.relu(self.a1(sa))
        a1 = F.relu(self.a2(a1))
        a1 = self.a3(a1)
        return v1, a1


class TD3_SRDC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=3e-4,
            critic_lr=3e-4,
            adaptive=True,
            alpha_SRDC=0.9,
            num_sample=5
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim,num_sample).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.adaptive = adaptive
        self.alpha = alpha_SRDC

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            mu=self.actor(state)

            # Compute the target Q value
            target_Q1, target_Q2, _, _ = self.critic_target(next_state, next_action, False, mu, self.expl_noise, self.max_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2, current_mean1, current_mean2 = self.critic(state, action, True, mu, self.expl_noise, self.max_action)
        baseline = torch.zeros_like(current_mean1)
        # Compute critic loss
        Q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        regular_loss = F.mse_loss(current_mean1, baseline) + F.mse_loss(current_mean2, baseline)
        if self.adaptive:
            with torch.no_grad():
                alpha = Q_loss/(Q_loss + regular_loss)
                beta = 1 - alpha
            critic_loss = alpha * Q_loss + beta * regular_loss
        else:
            critic_loss = self.alpha * Q_loss + (1-self.alpha) * regular_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            v, a = self.critic.Q1(state, self.actor(state))
            v = float(v.mean())
            actor_loss = - a.mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return Q_loss, regular_loss, v, -actor_loss, True
        return Q_loss, regular_loss, 0, 0, False

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
