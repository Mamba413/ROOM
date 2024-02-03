import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import d4rl
import numpy as np
import torch
import itertools
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import resample

from sacN import SAC_N
from util import set_seed, Log, sample_batch, evaluate_policy
from util import get_env_and_heavy_tail_dataset

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="halfcheetah-medium-v2", 
                    help='Mujoco Gym environment (default: halfcheetah-medium-v2)')
parser.add_argument('--policy', type=str, default='MM', 
                    help='BENCH / MM / BM')
parser.add_argument('--aggregate', type=str, 
                    default='Quantile', help="Quantile / MeanMStd")
parser.add_argument('--agent-policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, 
                    metavar='G', help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=3000001, metavar='N',
                    help='maximum number of steps (default: 3000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, 
                    metavar='N', help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--eval-period', type=int, default=5000)
parser.add_argument('--n-eval-episodes', type=int, default=10)
parser.add_argument('--max-episode-steps', type=int, default=1000)
parser.add_argument('--log-dir', type=str, default='NONE')
parser.add_argument("--ensemble", default=10, type=int)
parser.add_argument("--k-fold", default=10, type=int)
parser.add_argument('--quantile', type=float, default=0.5)
parser.add_argument('--scale', type=float, default=2.0)
parser.add_argument('--df', type=float, default=1.0)
parser.add_argument('--std-noise', type=float, default=0.0)
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: True)')
args = parser.parse_args()

if __name__ == '__main__':
    if args.policy == "BENCH":
        K_FOLDS = 1
    else:
        K_FOLDS = args.k_fold
    
    if args.log_dir == "NONE":
        args.log_dir = args.policy

        if args.aggregate == "Quantile":
            args.log_dir = args.log_dir + "_q_{}".format(args.quantile)
        elif args.aggregate == "MeanMStd":
            args.log_dir = args.log_dir + "_std_{}".format(args.scale)
        
    log = Log(Path(args.log_dir)/args.env_name/'seed_{}'.format(args.seed), vars(args))
    log(f'Log dir: {log.dir}')

    # Environment and dataset
    env, dataset = get_env_and_heavy_tail_dataset(log, args.env_name, args.max_episode_steps, 12345*args.seed, df=args.df, std_noise=args.std_noise, TD3BC=True)
    obs_dim = dataset['observations'].shape[1]
    set_seed(args.seed, env=env)

    # Agent
    agent = SAC_N(obs_dim, env.action_space, args)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, agent, args.max_episode_steps) for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    # Dataset
    num = dataset['observations'].shape[0]
    dataset_list = []
    if K_FOLDS > 1:
        if args.policy == "MM":
            group = np.array(list(itertools.chain.from_iterable([
                [i] * traj_len for i, traj_len in enumerate(np.ediff1d(np.nonzero(dataset['terminals'].cpu()).flatten()))
                ])))
            group = np.append(group, np.array([len(np.unique(group)) + 1] * (num - len(group))))
            if K_FOLDS < len(np.unique(group)):
                kf = GroupKFold(n_splits=K_FOLDS)
                for train_index, test_index in kf.split(np.zeros((num, 1)), groups=group):
                    tmp_dataset = {}
                    for key in list(dataset.keys()):
                        tmp_dataset[key] = dataset[key][test_index]
                        pass
                    dataset_list.append(tmp_dataset)
            else:
                kf = KFold(n_splits=K_FOLDS)
                for train_index, test_index in kf.split(np.zeros((num, 1))):
                    tmp_dataset = {}
                    for key in list(dataset.keys()):
                        tmp_dataset[key] = dataset[key][test_index]
                        pass
                    dataset_list.append(tmp_dataset)
        elif args.policy == "BM":
            for k in range(K_FOLDS):
                train_index = resample(np.arange(num))
                tmp_dataset = {}
                for key in list(dataset.keys()):
                    tmp_dataset[key] = dataset[key][train_index]
                    pass
                dataset_list.append(tmp_dataset)

    # Update agent
    for step in range(args.num_steps):
        if args.policy == "BENCH":
            agent.update_parameters(updates=step, **sample_batch(dataset, args.batch_size))
        else:
            for i in range(args.ensemble):
                agent.update_critic_i_parameters(i = i, updates=step, **sample_batch(dataset_list[i], args.batch_size))
            agent.update_actor_parameters(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            eval_policy()

    env.close()
    log.close()