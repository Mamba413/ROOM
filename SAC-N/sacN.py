import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, MLP, DeterministicPolicy

class SAC_N(object):
    def __init__(self, num_inputs, action_space, args):

        self.quantile = args.quantile
        self.ensemble = args.ensemble
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.agent_policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        def min_value(input):
            return torch.min(input, 1).values
        def quantile_value(input):
            return torch.quantile(input, q=args.quantile, dim=1)
        def mean_minus_std_value(input):
            return torch.mean(input, dim=1) - args.scale * torch.std(input, dim=1)
        if args.policy == "BENCH" and args.aggregate == "Quantile":
            self.value_fun = min_value
        elif args.policy == "BENCH" and args.aggregate == "MeanMStd":
            self.value_fun = mean_minus_std_value
        elif args.aggregate == "Quantile":
            self.value_fun = quantile_value
        elif args.aggregate == "MeanMStd":
            self.value_fun = mean_minus_std_value
        else:
            pass

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = [MLP(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device) for _ in range(self.ensemble)]
        self.critic_optim = [Adam(single_critic.parameters(), lr=args.lr) for single_critic in self.critic]

        self.critic_target = [MLP(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device) for _ in range(self.ensemble)]
        [hard_update(self.critic_target[i], self.critic[i]) for i in range(self.ensemble)]

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = state.unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, updates, observations, actions, rewards, next_observations, terminals):
        # print("rewards: ", rewards.shape)
        # update critics:
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_observations)
            # print("next_state_action: ", next_state_action.shape)
            qf_next_target = torch.cat([one_critic_target(next_observations, next_state_action) for one_critic_target in self.critic_target], 1)
            # print("qf_next_target: ", qf_next_target.shape)
            min_qf_next_target = self.value_fun(qf_next_target) - self.alpha * torch.flatten(next_state_log_pi)
            # print("min_qf_next_target: ", min_qf_next_target.shape)
            next_q_value = rewards + (1 - terminals.int()) * self.gamma * (min_qf_next_target)
            # print("next_q_value", next_q_value.shape)

        for i in range(self.ensemble):
            qf_i = torch.flatten(self.critic[i](observations, actions))
            qf_i_loss = F.mse_loss(qf_i, next_q_value)  
            self.critic_optim[i].zero_grad()
            qf_i_loss.backward()
            self.critic_optim[i].step()

        # update actor:
        pi, log_pi, _ = self.policy.sample(observations)
        qf = torch.cat([one_critic(observations, pi) for one_critic in self.critic], 1)
        min_qf_pi = self.value_fun(qf)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # update temperature:
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        # update critic target:
        if updates % self.target_update_interval == 0:
            [soft_update(self.critic_target[i], self.critic[i], self.tau) for i in range(self.ensemble)]

    def update_critic_i_parameters(self, i, updates, observations, actions, rewards, next_observations, terminals):
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_observations)
            qf_next_target = torch.cat([one_critic_target(next_observations, next_state_action) for one_critic_target in self.critic_target], 1)
            qf_next_target = self.value_fun(qf_next_target) - self.alpha * torch.flatten(next_state_log_pi)
            next_q_value = rewards + (1 - terminals.int()) * self.gamma * (qf_next_target)

        qf_i = torch.flatten(self.critic[i](observations, actions))
        qf_i_loss = F.mse_loss(qf_i, next_q_value)  
        self.critic_optim[i].zero_grad()
        qf_i_loss.backward()
        self.critic_optim[i].step()
        
        # update critic target:
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target[i], self.critic[i], self.tau)

    def update_actor_parameters(self, observations, actions, rewards, next_observations, terminals):
        pi, log_pi, _ = self.policy.sample(observations)
        qf = torch.cat([one_critic(observations, pi) for one_critic in self.critic], 1)
        qf_pi = self.value_fun(qf)

        policy_loss = ((self.alpha * log_pi) - qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # update temperature:
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()