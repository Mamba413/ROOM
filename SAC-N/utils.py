import math
import torch

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_ensemble_update(target_list, source_list, tau, ensemble):
    for i in range(ensemble):
        for target_param, param in zip(target_list[i].parameters(), source_list[i].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_ensemble_update(target_list, source_list, ensemble):
    for i in range(ensemble):
        for target_param, param in zip(target_list[i].parameters(), source_list[i].parameters()):
            target_param.data.copy_(param.data)

def evaluate_policy(env, agent, max_episode_steps):
    state = env.reset()
    episode_reward = 0.
    i = 1
    done = False
    while not done or i <= max_episode_steps:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        i = i + 1
    
    return episode_reward