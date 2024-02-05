# %%
import _MM_OPO as MM_OPO
import _PB_OPO as PB_OPO
import _rllib_util
from ray.rllib.agents import ppo
import ray
import _Simulator_cartpole as _Simulator
import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
from _util import *
from IPython.display import display

####################################################################################
gpu_number = 7
reload(_Simulator)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ray.init(num_gpus=1, num_cpus=2)
# from ray.rllib.algorithms.ppo import PPO
reload(MM_OPO)
reload(PB_OPO)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %%
n_gpu = 8

N = 400
T = 100
gamma = .99
A_range = [0, 1]
pess_quantile = 0.4

# %%
heavy_tailed = True
std_noise = 0.01  # state
##
alpha_levy = 0.8
df = 1
heavy_tail = 't'
L = 5

# %%
#################################
N_eval = 5000
batch_size = 256
max_iter = 100
eps = 0.005
test_freq = 30
verbose = 0
###
rep = 64
run_internal_Q_MM = True
run_internal_Q_PB = True
PRINT_FREQ = 4

# %%
env_config = dict(
    std=std_noise,
    seed=42,
    heavy_tailed=heavy_tailed,
    heavy_tail=heavy_tail,
    df=df,
    alpha_levy=alpha_levy,
    reward_noise_std=0,  
)
# used only to initialize the RL agent [no any real meaning]
agent_config = _rllib_util.get_agent_config(env_config, lr=0.001, gamma=gamma, n_hidden=32)

# %%
reload(_Simulator)
reload(MM_OPO)
reload(PB_OPO)


class Runner():
    @autoargs()
    def __init__(self, env_config, agent_config):
        pass

    def simulate_data(self, epsilon_behav, df, gamma, std_noise):
        ########### Simulate Data ###########
        simulator = _Simulator.Simulator(self.env_config)
        #### Optimal policy
        trainer = ppo.PPOTrainer(env=_Simulator.Simulator, config=self.agent_config)
        trainer.restore('cp/cartpole_gamma_{}/checkpoint_000020/checkpoint-20'.format(gamma))
        pi = RLlibPolicy(trainer)
        ####
        behav = EpsilonPolicy(pi, epsilon=epsilon_behav, n_actions=len(A_range))

        trajs_train_all = simulator.simu_trajs_parallel(pi=behav,  # eps_policy #None
                                                        seed=42, N=N * rep, T=T)
        trajs_train_resp = [trajs_train_all[(i * N):((i + 1) * N)] for i in range(rep)]
        printR("Simulate Data DONE!")
        return trajs_train_resp, behav, pi

    def compute_base_values(self, epsilon_behav, df, gamma, std_noise, behav, pi):
        ########### Prepare evaluation env ###########
        # since we care about the mean reward, during evaluation, no need to have error
        self.env_config['reward_noise_std'] = 0
        simulator_eval = _Simulator.Simulator(self.env_config)
        V_optimal = simulator_eval.evaluate_policy(pi=pi, gamma=gamma, init_states=None, N=N_eval, T=T, seed=42)
        V_behav = simulator_eval.evaluate_policy(pi=behav, gamma=gamma, init_states=None, N=N_eval, T=T, seed=42)
        printR('V_optimal = {:.2f}, V_behav = {:.2f}'.format(V_optimal, V_behav))
        return V_optimal, V_behav

    def run_simu(self, epsilon_behav, df, gamma, std_noise, trajs_train_resp):
        def one_seed(seed, trajs_train):
            gpu_idx = seed % n_gpu

            ########### Learn MM Policies ###########
            are = MM_OPO.MMOPO(trajs_train, gamma=gamma,
                               gpu_number=gpu_idx, L=L, seed=seed, 
                               A_range=A_range, pess_quantile=pess_quantile)
            are.learn_policies(max_iter=max_iter, run_internal_Q_MM=run_internal_Q_MM,
                               batch_size=batch_size, eps=eps, test_freq=test_freq, verbose=verbose)

            ########### Learn PB Policies ###########
            pbo = PB_OPO.PBOPO(trajs_train, gamma=gamma,
                               gpu_number=gpu_idx, L=L, seed=seed, 
                               A_range=A_range, pess_quantile=pess_quantile)
            pbo.learn_policies(max_iter=max_iter, run_internal_Q_PB=run_internal_Q_PB,
                               batch_size=batch_size, eps=eps, test_freq=test_freq, verbose=verbose)

            ########### Evaluate Policies ###########
            simulator_eval = _Simulator.Simulator(env_config)
            res1 = {
                policy_name: simulator_eval.evaluate_policy(pi=are.policies[policy_name],
                                                            gamma=gamma, init_states=None,
                                                            N=N_eval, T=T, seed=seed, print_out=False)
                for policy_name in are.policies
            }
            res2 = {
                policy_name: simulator_eval.evaluate_policy(pi=pbo.policies[policy_name],
                                                            gamma=gamma, init_states=None,
                                                            N=N_eval, T=T, seed=seed, print_out=False)
                for policy_name in pbo.policies
            }
            res = {**res1, **res2}

            if seed % PRINT_FREQ == 0:
                print("The {} replications done!".format(seed))
                print(res)

            return res

        ########### Run ###########
        res = []
        res = [one_seed(seed, trajs_train_resp[seed]) for seed in range(rep)]

        df_res = DF([DF(res).mean(0), DF(res).median(0), DF(res).std(0) / np.sqrt(len(res))], 
                    index=['mean', 'median', 'std_of_mean'])
        return df_res, res

    def run_one_setting(self, epsilon_behav, df, gamma, std_noise):
        print('epsilon_behav = {}, df = {}, gamma = {}, std_noise = {}'.format(
            epsilon_behav, df, gamma, std_noise))
        self.env_config['df'] = df
        self.env_config['reward_noise_std'] = std_noise
        self.trajs_train_resp, behav, pi = self.simulate_data(
            epsilon_behav, df, gamma, std_noise)
        V_optimal, V_behav = self.compute_base_values(
            epsilon_behav, df, gamma, std_noise, behav, pi)
        df_res, res = self.run_simu(
            epsilon_behav, df, gamma, std_noise, self.trajs_train_resp)
        # display(df_res)

        res_this_setting = {'df_res': df_res, 'details': res, 'V_optimal': V_optimal}
        return res_this_setting


# %% empirical performance when df increases (Figure 3(b))

res_settings = {}
ct = now()
##
path = "res/" + "cartpole" + "_opt" + "_T_{}".format(T) + "_N_{}".format(N) + "_" + EST()[7:9] + EST()[3:5] + EST()[:2] + EST()[3:5]
printR(path)

df_list = np.logspace(0, 1, num=5, base=2).tolist()
epsilon_behav_list = [0.05]

runner = Runner(env_config, agent_config)
for gamma in [0.99]:
    for std_noise in [0.5, 1.0, 2.0]:  
        res_settings[(gamma, std_noise)] = {}
        for epsilon_behav in epsilon_behav_list:
            for df in df_list:
                res_this_setting = runner.run_one_setting(epsilon_behav, df, gamma, std_noise)
                res_settings[(gamma, std_noise)][(epsilon_behav, df)] = res_this_setting
                dump(res_settings, path)
                clear_output()
                for a in res_settings:
                    gamma, std_noise = a
                    for b in res_settings[a]:
                        epsilon_behav, df = b
                        df_res = res_settings[a][b]['df_res']
                        print('epsilon_behav = {}, df = {}, gamma = {}, std_noise = {}'.format(
                            epsilon_behav, df, gamma, std_noise))
                        display(df_res)
                print((now() - ct) // 60, "min")
                printR(path)
