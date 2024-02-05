import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100000000.


def sql_loss(adv, v, alpha):
    scaled_adv = adv / (2 * alpha)
    return torch.mean((scaled_adv > -1).float() * (1 + scaled_adv)**2 + v / alpha)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, ivr_alpha, discount=0.99, alpha=0.005, objective="SQL"):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        if not hasattr(qf, "ensemble"):
            self.q_optimizer = optimizer_factory(self.qf.parameters())
        if not hasattr(vf, "ensemble"):
            self.v_optimizer = optimizer_factory(self.vf.parameters())
        if policy is not None:
            self.policy = policy.to(DEFAULT_DEVICE)
            self.policy_optimizer = optimizer_factory(self.policy.parameters())
            self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.ivr_alpha = ivr_alpha
        self.discount = discount
        self.alpha = alpha
        self.objective=objective

    def update_q_func(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        if self.objective == "IQL":
            v_loss = asymmetric_l2_loss(adv, self.tau)
        elif self.objective == "SQL":
            v_loss = sql_loss(adv, v, self.ivr_alpha)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        with torch.no_grad():
            next_v = self.vf(next_observations)
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

    def update_q_func_MM_internal(self, observations, actions, next_observations, rewards, terminals, mmq, mmv):
        with torch.no_grad():
            target_q = mmq(observations, actions)
            next_v = mmv(next_observations)

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

    def update_policy(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            v = self.vf(observations)

        adv = (target_q - v).to(DEFAULT_DEVICE)
        if self.objective == "IQL":
            adv_func = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        elif self.objective == "SQL":
            adv_func = (1 + adv.detach() / (2 * self.ivr_alpha)).clamp(min = 0.0)
        selected_update_idx = torch.where(~torch.isinf(adv_func))[0]
        adv_func = adv_func[selected_update_idx]
        
        policy_out = self.policy(observations[selected_update_idx])
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions[selected_update_idx])
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out[selected_update_idx] - actions[selected_update_idx])**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(adv_func * bc_losses)
            
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()