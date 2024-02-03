import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys
import gym
import d4rl
import h5py

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import levy_stable, t

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


def return_q_range(dataset, max_episode_steps, gamma):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += np.power(gamma, ep_len) * float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    
    q_value = np.array(returns)
    print("Q range: [{}, {}] with mean {}, std {}".format(
        np.min(q_value), 
        np.max(q_value),
        np.mean(q_value), 
        np.std(q_value),
    ))
    
    return min(returns), max(returns)

# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}


def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        with torch.no_grad():
            action = policy.select_action(torchify(obs), evaluate=deterministic)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward

def evaluate_policy2(env, policy, max_episode_steps, mean, std, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        obs = (np.array(obs).reshape(1,-1) - mean)/std
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward

def rollout_trajectory(env, policy, max_episode_steps, deterministic=True):
    obs = env.reset()
    obs_list = []
    action_list = []
    next_obs_list = []
    reward_list = []
    terminal_list = []
    for _ in range(max_episode_steps):
        obs_list.append(obs)
        with torch.no_grad():
            action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        action_list.append(action)
        next_obs, reward, done, info = env.step(action)
        next_obs_list.append(next_obs)
        reward_list.append(reward)
        terminal_list.append(done)
        if done:
            break
        else:
            obs = next_obs
    return [obs_list, action_list, next_obs_list, reward_list, terminal_list]

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()

def get_env_and_dataset_no_log(env_name, max_episode_steps, gamma):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.
    print('Rewards have range [{}, {}] with mean {}'.format(
        np.min(dataset['rewards']), 
        np.max(dataset['rewards']), 
        np.mean(dataset['rewards']),
    ))
    return_q_range(dataset, max_episode_steps, gamma)

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset     

def get_env_and_dataset(log, env_name, max_episode_steps, torchify_data=True, TD3BC=False):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if not TD3BC:
        if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
            min_ret, max_ret = return_range(dataset, max_episode_steps)
            log(f'Dataset returns have range [{min_ret}, {max_ret}]')
            dataset['rewards'] /= (max_ret - min_ret)
            dataset['rewards'] *= max_episode_steps
        elif 'antmaze' in env_name:
            dataset['rewards'] -= 1.

    if torchify_data:
        for k, v in dataset.items():
            dataset[k] = torchify(v)

    return env, dataset            

def gen_noise(size, seed, heavy_tail, noise_std, **kwargs):
    # noise_std = kwargs['noise_std']
    # seed = kwargs['noise']
    # size = kwargs['size']
    # heavy_tail = kwargs['heavy_tail']
    if heavy_tail == 'levy':
        #rvs = levy_stable.rvs(alpha = self.alpha_levy, beta = 0, loc=0, scale=1, size = 10000, random_state=0)
        rvs = levy_stable.rvs(size=10000, random_state=42, **kwargs)
    elif heavy_tail == 't':
        rvs = t.rvs(size=10000, random_state=42, **kwargs)
    """ 
    When heavy-tailed, the variance may not exist
    If we scale by varaince, for most entries, if might be very small, which is not acceptable?
    For a quick fix, I will clip and then compute the variance
    """
    clipped_rvs = np.clip(rvs, np.quantile(rvs, [0.01]), np.quantile(rvs, [0.99]))
    sample_std = np.std(clipped_rvs)
    
    if heavy_tail == 'levy':
        rvs = levy_stable.rvs(size=size, random_state=seed, **kwargs)
    elif heavy_tail == 't':
        rvs = t.rvs(size=size, random_state=seed, **kwargs)

    rvs = rvs / sample_std * noise_std 
    return rvs


def get_env_and_heavy_tail_dataset(log, env_name, max_episode_steps, seed, torchify_data=True, TD3BC=False, df=1.0, std_noise=1.0):
    is_diy_data = False
    if env_name.split("-")[0] == "deterministic":
        is_diy_data = True
        env = gym.make(env_name.replace("deterministic-", ""))
    elif env_name.split("-")[0].startswith("epsilon"):
        env = gym.make(env_name.replace(env_name.split("-")[0]+'-', ""))
    else:
        env = gym.make(env_name)
    
    if is_diy_data:
        filename = "/Users/bytedance/Documents/MM-RL/offline-data/{}.hdf5".format(env_name)
        with h5py.File(filename, "r") as f:
            group_key_list = [
                'observations', 'actions', 'next_observations','rewards', 'terminals'
            ]
            dataset = {
                a_group_key: f[a_group_key][()] for a_group_key in group_key_list
            }
        f.close()
    else:
        dataset = d4rl.qlearning_dataset(env)
    
    num = dataset['observations'].shape[0]
    t_dist_param = {
        'df': df,
    }

    # dataset['rewards'] += gen_noise(size=num, seed=seed, heavy_tail='t', noise_std=std_noise, **t_dist_param)   # no significant difference
    if not TD3BC:
        if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
            min_ret, max_ret = return_range(dataset, max_episode_steps)
            log(f'Dataset returns have range [{min_ret}, {max_ret}]')
            ret_range_gap = max_ret - min_ret
            dataset['rewards'] /= ret_range_gap
            dataset['rewards'] *= max_episode_steps
        elif 'antmaze' in env_name:
            dataset['rewards'] -= 1.

    noise_value = gen_noise(size=num, seed=seed, heavy_tail='t', noise_std=std_noise, **t_dist_param)
    print("noise mean: {:.2f}; std: {:.2f}".format(np.mean(noise_value), np.std(noise_value)))
    dataset['rewards'] += noise_value
    
    if torchify_data:
        for k, v in dataset.items():
            dataset[k] = torchify(v)

    return env, dataset


def gen_noise_action(size, action, seed, heavy_tail, **kwargs):
    if heavy_tail == 'levy':
        rvs = levy_stable.rvs(size=10000, random_state=42, **kwargs)
    elif heavy_tail == 't':
        rvs = t.rvs(size=10000, random_state=42, **kwargs)
    clipped_rvs = np.clip(rvs, np.quantile(rvs, [0.01]), np.quantile(rvs, [0.99]))
    sample_std = np.std(clipped_rvs)
    
    if heavy_tail == 'levy':
        rvs = levy_stable.rvs(size=size, random_state=seed, **kwargs)
    elif heavy_tail == 't':
        rvs = t.rvs(size=size, random_state=seed, **kwargs)

    action_norm = np.linalg.norm(action, axis=1)    
    rvs = rvs / sample_std * action_norm 
    return rvs


def get_env_and_heavy_tail_byaction_dataset(log, env_name, max_episode_steps, seed, torchify=True):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    
    num = dataset['observations'].shape[0]
    t_dist_param = {
        'df': 3.0,
    }

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        ret_range_gap = max_ret - min_ret
        dataset['rewards'] /= ret_range_gap
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    dataset['rewards'] += gen_noise_action(size=num, action=dataset['actions'], seed=seed, heavy_tail='t', **t_dist_param)
    
    if torchify:
        for k, v in dataset.items():
            dataset[k] = torchify(v)

    return env, dataset

