# %%
import ray
import gym
import _rllib_util
from ray.rllib.agents import ppo
import _MM_OPO as _MM_OPO
import _MM_OPE as _MM_OPE
from IPython import display
from _util import *
import _Simulator_cartpole as _Simulator
import _RL.FQE as FQE_module
import _RL.FQI as FQI
import _RL.my_gym as my_gym
from IPython.display import display

reload(_Simulator)
##
reload(_MM_OPE)
reload(_MM_OPO)
##
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
n_gpu = 8

N = 100
T = 100
# [T longer ] We set T = 300 and γ = 0.98 for CartPole, and T = 200 and γ = 0.95 for Diabetes
gamma = .99
A_range = [0, 1]

# %%
heavy_tailed = True
std_noise = 0.01  # state
##
alpha_levy = 0.8
df = 1
heavy_tail = 't'
# heavy_tail = 'levy'
###
L = 5
BOOTSTRAP=False
INTERNAL_Q_MM = True
AVERAGE_MEASURE = True

# %%
N_EVAL = 5000
batch_size = 256
max_iter = 100
eps = 0.005
test_freq = 30
verbose = 0
###
rep = 64

# %%
optimal_policies = {}
for gamma in [0.99]:
    env_config = dict(std=std_noise, seed=42, heavy_tailed=heavy_tailed, heavy_tail=heavy_tail, df=df, alpha_levy=alpha_levy, reward_noise_std=0
                      )
    # used only to initialize the RL agent [no any real meaning]
    agent_config = _rllib_util.get_agent_config(
        env_config, lr=0.001, gamma=gamma, n_hidden=32)
    #### Optimal policy
    trainer = ppo.PPOTrainer(env=_Simulator.Simulator, config=agent_config)
    trainer.restore('cp/cartpole_gamma_{}/checkpoint_000100/checkpoint-100'.format(gamma))
    pi = RLlibPolicy(trainer)
    behav = EpsilonPolicy(pi, epsilon=0.3, n_actions=len(A_range))

    ###### Data ######
    reload(_Simulator)
    simulator = _Simulator.Simulator(env_config)

    # if N is too large, then... very slow
    trajs_train = simulator.simu_trajs_parallel(pi=behav, seed=42, N=1000, T=T)
    ###########################################################################
    ###### Training
    # mean reward, so can be learned in a noiseless fashion
    #gamma_train = 0.5
    ######
    N_EVAL = 20000
    gpu_idx = 0
    FQI_paras = {
        "hiddens": [32, 32],  # 32 seems good enough
        "gamma": gamma, 
        'num_actions': 2, 
        'batch_size': batch_size, 
        'eps': 0.005
    }
    verbose = 1
    test_freq = 10
    ###
    import _RL.FQI as FQI
    reload(FQI)
    naive_FQI_policy = FQI.FQI(max_iter=max_iter, gpu_number=gpu_idx, **FQI_paras)
    naive_FQI_policy.train(trajs=trajs_train, train_freq=test_freq, verbose=verbose,
                           nn_verbose=0, validation_freq=1, path=None, save_freq=10, es=False)

    ###### Evaluation
    # mean reward, so not related to reward error (i.e., std = 0)
    # only gamma matters
    ######
    simulator = _Simulator.Simulator(env_config)
    V = simulator.evaluate_policy(
        pi=naive_FQI_policy, gamma=gamma, init_states=None, N=N_EVAL, T=T)
    print(V)
    optimal_policies[gamma] = {'policy': naive_FQI_policy, 'value': V}

# %%
trajs = trajs_train[:20]
S, A, R, SS = [np.array([item[i] for traj in trajs for item in traj])
               for i in range(4)]
dataset = {
    'observations': S,
    'next_observations': SS,
    'actions': A,
    'rewards': R,
    'terminals': np.zeros(len(R))
}
dump(dataset, 'dataset')

# %%
#################################
N_EVAL = 5000
T_EVAL = 300
batch_size = 256
###
max_iter = 100
eps = 0.005
test_freq = 30
verbose = 0
L = 5

# %%
reload(_Simulator)
reload(_MM_OPO)


class Runner():
    @autoargs()
    def __init__(self, env_config, agent_config):
        pass

    def simulate_data(self, epsilon_behav, df, gamma, std_noise):
        ########### Simulate Data ###########
        simulator = _Simulator.Simulator(self.env_config)
        #### Optimal policy
        pi = optimal_policies[gamma]['policy']
        behav = EpsilonPolicy(pi, epsilon=epsilon_behav, n_actions=len(A_range))

        trajs_train_all = simulator.simu_trajs_parallel(pi=behav, seed=42, N=N*rep, T=T)
        trajs_train_resp = [
            trajs_train_all[(i * N):((i + 1) * N)] for i in range(rep)]
        init_S = simulator.simu_init_S(seed=42, N=1000)
        printR("Simulate Data DONE!")
        return trajs_train_resp, behav, pi, init_S

    def compute_base_values(self, epsilon_behav, df, gamma, std_noise, behav, pi):
        ########### Prepare evaluation env ###########
        # since we care about the mean reward, during evaluation, no need to have error
        self.env_config['reward_noise_std'] = 0
        simulator_eval = _Simulator.Simulator(self.env_config)
        V_optimal = simulator_eval.evaluate_policy(pi=pi, gamma=gamma, init_states=None, N=N_EVAL, T=T_EVAL, seed=42)
        V_behav = simulator_eval.evaluate_policy(pi=behav, gamma=gamma, init_states=None, N=N_EVAL, T=T_EVAL, seed=42)
        printR('V_optimal = {:.2f}, V_behav = {:.2f}'.format(V_optimal, V_behav))
        return V_optimal, V_behav

    def run_simu(self, epsilon_behav, df, gamma, std_noise, trajs_train_resp, pi, V_optimal, init_S):
        FQE_paras = {"hiddens": [32, 32],  # 32 seems good enough
                     "gamma": gamma, 'num_actions': 2, 'batch_size': batch_size, 'eps': 0.005, 'max_iter': max_iter}

        def one_seed(seed, trajs_train):
            gpu_idx = seed % n_gpu

            are = _MM_OPE.ARE(trajs_train, pi=pi, gamma=gamma, gpu_number=gpu_idx, L=L, bootstrap=BOOTSTRAP)
            are.init_S = init_S  # a little large for distributed
            if BOOT_MM_COMPARISON:
                are.BM_MM_compare(**FQE_paras, test_freq=10, verbose=0)
            else:
                are.est_Q(**FQE_paras, test_freq=10, verbose=0, run_internal_Q_MM=INTERNAL_Q_MM)
            
            if seed % PRINT_FREQ == 0:
                print("The {} replications done!".format(seed))
                print(are.values)
                
            return are.values

        res = [one_seed(seed, trajs_train_resp[seed]) for seed in range(rep)]

        reps = len(res)
        if AVERAGE_MEASURE:
            MSE = ((DF(res) - V_optimal) ** 2).mean(0)
            MSE_std = ((DF(res) - V_optimal) ** 2).std(0) / np.sqrt(reps)
            df_res = DF([MSE, MSE_std], index=['MSE', 'MSE_std'])
        else:
            median_MSE = ((DF(res) - V_optimal) ** 2).median(0)
            median_MSE_std = np.max(np.abs(((DF(res) - V_optimal) ** 2).quantile([0.05, 0.95]) - median_MSE)) / np.sqrt(reps)
            df_res = DF([median_MSE, median_MSE_std], index=['median_MSE', 'median_MSE_std'])

        return df_res, res
    

    def run_tradeoff_simu(self, epsilon_behav, df, gamma, std_noise, trajs_train_resp, pi, V_optimal, init_S):
        FQE_paras = {"hiddens": [32, 32],  # 32 seems good enough
                     "gamma": gamma, 'num_actions': 2, 'batch_size': batch_size, 'eps': 0.005, 'max_iter': max_iter}

        def one_seed(seed, trajs_train):
            gpu_idx = seed % n_gpu

            are = _MM_OPE.ARE(trajs_train, pi=pi, gamma=gamma, gpu_number=gpu_idx, L=L, bootstrap=BOOTSTRAP)
            are.init_S = init_S  # a little large for distributed

            are.est_Q2(**FQE_paras, test_freq=10, verbose=0, run_internal_Q_MM=INTERNAL_Q_MM)
            
            if seed % PRINT_FREQ == 0:
                print("The {} replications done!".format(seed))
                print(are.values)
                
            return are.values, are.runtimes

        res = [one_seed(seed, trajs_train_resp[seed]) for seed in range(rep)]
        runtime = DF([x[1] for x in res]).mean(0)
        runtime_std = DF([x[1] for x in res]).std(0)
        res = [x[0] for x in res]
        reps = len(res)
        runtime_std = runtime_std / np.sqrt(reps)
        if AVERAGE_MEASURE:
            MSE = ((DF(res) - V_optimal) ** 2).mean(0)
            MSE_std = ((DF(res) - V_optimal) ** 2).std(0) / np.sqrt(reps)
            df_res = DF([MSE, MSE_std, runtime, runtime_std], 
                        index=['MSE', 'MSE_std', 'Runtime', 'Runtime_std'])
        else:
            median_MSE = ((DF(res) - V_optimal) ** 2).median(0)
            median_MSE_std = np.max(np.abs(((DF(res) - V_optimal) ** 2).quantile([0.05, 0.95]) - median_MSE)) / np.sqrt(reps)
            df_res = DF([median_MSE, median_MSE_std, runtime, runtime_std], 
                        index=['median_MSE', 'median_MSE_std', 'Runtime', 'Runtime_std'])

        return df_res, res

    def run_one_setting(self, epsilon_behav, df, gamma, std_noise, time_robust_tradeoff=False):
        print('epsilon_behav = {}, df = {}, gamma = {}, std_noise = {}'.format(epsilon_behav, df, gamma, std_noise))
        self.env_config['df'] = df
        self.env_config['reward_noise_std'] = std_noise
        trajs_train_resp, behav, pi, init_S = self.simulate_data(epsilon_behav, df, gamma, std_noise)
        V_optimal, V_behav = self.compute_base_values(epsilon_behav, df, gamma, std_noise, behav, pi)
        if time_robust_tradeoff:
            df_res, res = self.run_tradeoff_simu(epsilon_behav, df, gamma, std_noise, trajs_train_resp, pi, V_optimal, init_S)
        else:
            df_res, res = self.run_simu(epsilon_behav, df, gamma, std_noise, trajs_train_resp, pi, V_optimal, init_S)

        res_this_setting = {
            'df_res': df_res, 'details': res,
            'V_optimal': V_optimal, 'V_behav': V_behav
        }

        return res_this_setting

# %% empirical performance when df increases (Figure 3(a))

BOOT_MM_COMPARISON = False
res_settings = {}
ct = now()
PRINT_FREQ = 32
path = "res/" + "cartpole" + "_eval" + "_T_{}".format(T) + "_N_{}".format(N) + "_" + EST()[7:9] + EST()[3:5] + EST()[:2] + EST()[3:5]
printR(path)
runner = Runner(env_config, agent_config)
STD_LIST = [1, 2]
EPSILON_LIST = [0.05]
DF_LIST = np.logspace(0.0, 0.6, 5, base=2.0).tolist()
for gamma in [0.99]:  
    for std_noise in STD_LIST:
        res_settings[(gamma, std_noise)] = {}
        for epsilon_behav in EPSILON_LIST:
            for df in DF_LIST:
                res_this_setting = runner.run_one_setting(epsilon_behav, df, gamma, std_noise)
                res_settings[(gamma, std_noise)][(epsilon_behav, df)] = res_this_setting
                ########### Save ###########
                #res_settings[(epsilon_behav, df, gamma, std_noise)] = res_this_setting
                dump(res_settings, path)
                ########### Analysis ###########
                clear_output()
                for a in res_settings:
                    gamma, std_noise = a
                    for b in res_settings[a]:
                        epsilon_behav, df = b
                        df_res = res_settings[a][b]['df_res']
                        print('epsilon_behav = {}, df = {}, gamma = {}, std_noise = {}'.format(
                            epsilon_behav, df, gamma, std_noise))
                        #df_res.iloc[0] = optimal_value[gamma] - df_res.iloc[0]
                        display(df_res)
                print((now() - ct) // 60, "min")
                printR(path)

# %% empirical comparison: bootstrap and MM (Figure 5)

res_settings = {}
BOOT_MM_COMPARISON = True
ct = now()
PRINT_FREQ = 32
path = "res/" + "cartpole" + "_eval" + "_T_{}".format(T) + "_N_{}".format(N) + "_" + EST()[7:9] + EST()[3:5] + EST()[:2] + EST()[3:5]
printR(path)
runner = Runner(env_config, agent_config)
STD_LIST = [1]
EPSILON_LIST = [0.05]
DF_LIST = np.logspace(0.0, 0.6, 5, base=2.0).tolist()
for gamma in [0.99]: 
    for std_noise in STD_LIST:  
        res_settings[(gamma, std_noise)] = {}
        for epsilon_behav in EPSILON_LIST:
            for df in DF_LIST:
                res_this_setting = runner.run_one_setting(epsilon_behav, df, gamma, std_noise)
                res_settings[(gamma, std_noise)][(epsilon_behav, df)] = res_this_setting
                ########### Save ###########
                dump(res_settings, path)
                ########### Analysis ###########
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
BOOT_MM_COMPARISON = False

# %% Trade-off study for robustness and runtime (Figure 6)

res_settings = {}
ct = now()
PRINT_FREQ = 4
BOOT_MM_COMPARISON = False
INTERNAL_Q_MM = True

path = "res/" + "cartpole" + "_eval_compute_robust_tradeoff" + "_T_{}".format(T) + "_N_{}".format(N) + "_" + EST()[7:9] + EST()[3:5] + EST()[:2] + EST()[3:5]
runner = Runner(env_config, agent_config)
EPSILON_LIST = [0.05]
DF_LIST = [1.5]
L_LIST = [11, 9, 7, 5, 3]
for gamma in [0.99]:  
    for std_noise in [1.0]:  
        res_settings[(gamma, std_noise)] = {}
        for epsilon_behav in EPSILON_LIST:
            for df in DF_LIST:
                for L in L_LIST:
                    res_this_setting = runner.run_one_setting(epsilon_behav, df, gamma, std_noise, time_robust_tradeoff=True)
                    res_settings[(gamma, std_noise)][(df, L)] = res_this_setting
                    dump(res_settings, path)
                    clear_output()
                    for a in res_settings:
                        gamma, std_noise = a
                        for b in res_settings[a]:
                            df, L = b
                            df_res = res_settings[a][b]['df_res']
                            print('L = {}, df = {}, gamma = {}, std_noise = {}'.format(L, df, gamma, std_noise))
                            display(df_res)
                    print((now() - ct) // 60, "min")
                    printR(path)
