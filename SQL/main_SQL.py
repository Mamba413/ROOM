import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import glob

from pathlib import Path
import copy

import itertools
import gym
import d4rl
import numpy as np
import torch
from tqdm import trange
from sklearn.model_selection import KFold, GroupKFold
from sklearn.utils import resample

from src.sql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction, MM_Q, MM_V, PB_Q, PB_V, Mean_Q, Mean_V
from src.util import set_seed, Log, sample_batch, evaluate_policy
from src.util import get_env_and_heavy_tail_dataset

def main(args):
    if args.policy == "BENCH":
        K_FOLDS = 1
    else:
        K_FOLDS = args.k_fold

    if args.log_dir == "NONE":
        args.log_dir = args.policy + "_{}".format(args.objective)

        if args.objective == "IQL":
            param_settings = "_tau_{}".format(args.tau)
        elif args.objective == "SQL":
            param_settings = "_IVRalpha_{}".format(args.ivr_alpha)
        args.log_dir = args.log_dir + param_settings

        if args.policy != "BENCH":
            if args.aggregate == "Quantile":
                args.log_dir = args.log_dir + "_q_{}".format(args.quantile)
            elif args.aggregate == "MeanMStd":
                args.log_dir = args.log_dir + "_std_{}".format(args.scale)

    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name/'seed_{}'.format(args.seed), vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_heavy_tail_dataset(log, args.env_name, args.max_episode_steps, 12345*args.seed, df=args.df, std_noise=args.std_noise)
    # env, dataset = get_env_and_heavy_tail_byaction_dataset(log, args.env_name, args.max_episode_steps, 12345*args.seed)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)
    
    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) for _ in range(args.n_eval_episodes)])
        if args.env_name.split("-")[0] == "deterministic":
            env_name = args.env_name.replace("deterministic-", "")
        elif args.env_name.split("-")[0].startswith("epsilon"):
            env_name = args.env_name.replace(args.env_name.split("-")[0]+"-", "")
        else:
            env_name = args.env_name
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })
    
    # Train multiple Q function:
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
    else:
        dataset_list.append(dataset)
            
    iql_list = []
    if args.policy == "BENCH":
        file_template = "./{}_{}{}/{}/seed_{}/**/final_{}.pt"
    else:
        file_template = "./{}_{}{}_**/{}/seed_{}/**/final_{}.pt"
    for k in range(K_FOLDS):
        iql = ImplicitQLearning(
            qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
            vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
            policy=None,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            max_steps=args.n_steps,
            tau=args.tau,
            beta=args.beta,
            ivr_alpha=args.ivr_alpha, 
            alpha=args.alpha,
            discount=args.discount, 
            objective=args.objective,
        )
        
        file_path = file_template.format(args.policy, args.objective, param_settings, args.env_name, args.seed, k)
        file_path = glob.glob(file_path)
        if len(file_path) == 0:
            dataset_k = dataset_list[k]
            for step in trange(args.n_steps):
                iql.update_q_func(**sample_batch(dataset_k, args.batch_size))
                if (step+1) % args.eval_period == 0:
                    print("Update {}-th Q function {} times".format(k, step+1))
            torch.save(iql.state_dict(), log.dir/'final_{}.pt'.format(k))
        else:
            log(f'load Q_{k} function from {file_path[0]}')
            iql.load_state_dict(torch.load(file_path[0]))

        iql_list.append(iql)
        pass
    
    q_list = [copy.deepcopy(x.q_target).requires_grad_(False) for x in iql_list]
    v_list = [copy.deepcopy(x.vf).requires_grad_(False) for x in iql_list]
    if args.aggregate == "MeanMStd":
        mmq = PB_Q(q_list, args.scale)
        mmv = PB_V(v_list, args.scale)
    elif args.aggregate == "Quantile":
        mmq = MM_Q(q_list, args.quantile)
        mmv = MM_V(v_list, args.quantile)
    else:
        pass

    iql_final = ImplicitQLearning(
        qf=mmq,
        vf=mmv,
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        ivr_alpha=args.ivr_alpha, 
        alpha=args.alpha,
        discount=args.discount,
        objective=args.objective,
    )    

    if args.policy_n_steps == 0:
        POLICY_UPDATE_TIME = args.n_steps
    else:
        POLICY_UPDATE_TIME = args.policy_n_steps

    for step in range(POLICY_UPDATE_TIME):
        iql_final.update_policy(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            eval_policy()

    torch.save(iql_final.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--policy', type=str, default='MM')          # BENCH / MM / BM
    parser.add_argument('--aggregate', type=str, default="Quantile") # Quantile / MeanMStd
    parser.add_argument('--env-name', type=str, default='halfcheetah-expert-v2')
    parser.add_argument('--objective', type=str, default="SQL")
    parser.add_argument('--log-dir', type=str, default='NONE')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--df', type=float, default=1.0)
    parser.add_argument('--std-noise', type=float, default=1.0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**5)
    parser.add_argument("--policy-n-steps", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--ivr-alpha', type=float, default=1.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument("--k-fold", default=5, type=int)
    parser.add_argument('--quantile', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=2.0)
    main(parser.parse_args())