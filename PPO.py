import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def dist_mode(self,state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v



class PPO(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		env_with_Dead,
		gamma=0.99,
		lambd=0.95,
		clip_rate=0.2,
		K_epochs=10,
		net_width=256,
		a_lr=3e-4,
		c_lr=3e-4,
		l2_reg = 1e-3,
		dist='Beta',
		a_optim_batch_size = 64,
		c_optim_batch_size = 64,
		entropy_coef = 0,
		entropy_coef_decay = 0.9998
	):
		if dist == 'Beta':
			self.actor = BetaActor(state_dim, action_dim, net_width).to(device)
		elif dist == 'GS_ms':
			self.actor = GaussianActor_musigma(state_dim, action_dim, net_width).to(device)
		elif dist == 'GS_m':
			self.actor = GaussianActor_mu(state_dim, action_dim, net_width).to(device)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
		self.dist = dist

		self.critic = Critic(state_dim, net_width).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

		self.env_with_Dead = env_with_Dead
		self.action_dim = action_dim
		self.clip_rate = clip_rate
		self.gamma = gamma
		self.lambd = lambd
		self.clip_rate = clip_rate
		self.K_epochs = K_epochs
		self.data = []
		self.l2_reg = l2_reg
		self.a_optim_batch_size = a_optim_batch_size
		self.c_optim_batch_size = c_optim_batch_size
		self.entropy_coef = entropy_coef
		self.entropy_coef_decay = entropy_coef_decay

	def select_action(self, state):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			dist = self.actor.get_dist(state)
			a = dist.sample()
			a = torch.clamp(a, 0, 1)
			logprob_a = dist.log_prob(a).cpu().numpy().flatten()
			return a.cpu().numpy().flatten(), logprob_a

	def evaluate(self, state):#only used when evaluate the policy.Making the performance more stable
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			if self.dist == 'Beta':
				a = self.actor.dist_mode(state)
			if self.dist == 'GS_ms':
				a,b = self.actor(state)
			if self.dist == 'GS_m':
				a = self.actor(state)
			return a.cpu().numpy().flatten(),0.0


	def train(self):
		self.entropy_coef*=self.entropy_coef_decay
		s, a, r, s_prime, logprob_a, done_mask, dw_mask = self.make_batch()


		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_prime)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs

			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()


	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst = [], [], [], [], [], [], []
		for transition in self.data:
			s, a, r, s_prime, logprob_a, done, dw = transition

			s_lst.append(s)
			a_lst.append(a)
			logprob_a_lst.append(logprob_a)
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			done_lst.append([done])
			dw_lst.append([dw])

		if not self.env_with_Dead:
			'''Important!!!'''
			# env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
			# env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
			dw_lst = (np.array(dw_lst)*False).tolist()

		self.data = [] #Clean history trajectory

		'''list to tensor'''
		with torch.no_grad():
			s, a, r, s_prime, logprob_a, done_mask, dw_mask = \
				torch.tensor(s_lst, dtype=torch.float).to(device), \
				torch.tensor(a_lst, dtype=torch.float).to(device), \
				torch.tensor(r_lst, dtype=torch.float).to(device), \
				torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
				torch.tensor(logprob_a_lst, dtype=torch.float).to(device), \
				torch.tensor(done_lst, dtype=torch.float).to(device), \
				torch.tensor(dw_lst, dtype=torch.float).to(device),


		return s, a, r, s_prime, logprob_a, done_mask, dw_mask


	def put_data(self, transition):
		self.data.append(transition)

	def save(self,episode):
		torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
		torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))


	def load(self,episode):
		self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
		self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))



