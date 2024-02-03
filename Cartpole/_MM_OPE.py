from _util import *

import _RL.FQE as FQE_module
import _RL.FQI as FQI
from _density import omega_SA, omega_SASA

reload(FQE_module)
reload(FQI)
reload(FQE_module)

import _RL.FQE_mm_internal as FQE_mm_internal
reload(FQE_mm_internal)

import _RL.sampler as sampler
# import GeoMedian_functional
# reload(GeoMedian_functional)
from time import time

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# tf.keras.backend.set_floatx('float64')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

################################################################################################################################################################################################################################################################################################################################################################################################################################################

class ARE():
    """ ADAPTIVE, EFFICIENT AND ROBUST OFF-POLICY EVALUATION
    for a dataset and a given policy
    , estimate the components (Q, omega, omega_star)
    , and construct the doubly, triply, ... robust estimators for the itegrated value
    """
    def __init__(self, trajs, pi, eval_N = 1000, gpu_number = 0, verbose = 0
                 , L = 2, incomplete_ratio = 20, sepe_A = 0, A_range = [0, 1, 2, 3, 4]
                 , gamma = .9, bootstrap=False):
        self.trajs = trajs # data: T transition tuple for N trajectories
        self.S, self.A, self.R, self.SS = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        self.N, self.T, self.L = len(trajs), len(trajs[0]), L
        self.S_dims = len(np.atleast_1d(trajs[0][0][0]))
        self.A_range = arr(A_range).astype(np.float64) #set(self.A)
        self.num_A = len(self.A_range)
        self.gamma = gamma
        self.pi = pi
        self.gpu_number = gpu_number
        self.alphas = alphas = [0.05, 0.1]
        self.z_stats = [sp.stats.norm.ppf((1 - alpha / 2)) for alpha in self.alphas]
        self.boot_split_ind = bootstrapping_sampling(self.L, self.N)
        self.split_ind = sample_split(self.L, self.N) # split_ind[k] = {"train_ind" : i, "test_ind" : j}
        self.eval_N = eval_N
        self.sepe_A = sepe_A
        self.verbose = verbose
        self.incomplete_ratio = incomplete_ratio
        
        self.value_funcs = []
        self.Q_values = {}
        self.trajs_splits = [[self.trajs[i] for i in self.split_ind[k]["test_ind"]]  for k in range(self.L)]
        self.boot_trajs_splits = [[self.trajs[i] for i in self.boot_split_ind[k]["test_ind"]]  for k in range(self.L)]
        self.complete_trajs = [self.trajs[i] for i in range(self.N)]
        self.raw_Qs = np.zeros(self.L)
        self.IS_it = np.zeros(self.L)
        self.trunc_IS_it = 0.0

    ############################################################################################################################
    ###########################################  The three main components #####################################################
    ############################################################################################################################
    def BM_MM_compare(self, verbose=1, test_freq=10, **FQE_paras):  
        self.values = {}
        ######### Estimate Q for every split  #########
        self.init_V_splits = {}
        value_func_mm_list = []
        value_func_bm_list = []
        for k in range(self.L):
            ################## MM type value ##################
            value_func_mm = FQE_module.FQE(policy = self.pi, 
                                           init_states = self.init_S,  # used for evaluation
                                           gpu_number = self.gpu_number, 
                                           **FQE_paras)
            train_traj = self.trajs_splits[k] #[self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            value_func_mm.train(train_traj, verbose = verbose,
                                # test_freq = test_freq
                                )
            init_V = value_func_mm.init_state_value(init_states = self.init_S)
            self.init_V_splits[k] = init_V
            V_mm = np.stack([self.init_V_splits[k] for k in self.init_V_splits], 1)
            ################## Bootstrap type value ##################
            value_func_bm = FQE_module.FQE(policy = self.pi, 
                                           init_states = self.init_S,  # used for evaluation
                                           gpu_number = self.gpu_number, 
                                           **FQE_paras)
            train_traj = self.trajs_splits[k] #[self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            value_func_bm.train(train_traj, verbose = verbose,
                                # test_freq = test_freq
                                )
            init_V = value_func_bm.init_state_value(init_states = self.init_S)
            self.init_V_splits[k] = init_V
            V_bm = np.stack([self.init_V_splits[k] for k in self.init_V_splits], 1)

            # baseline
            self.est_naive_FQE(verbose = verbose, train_freq = test_freq, **FQE_paras)
            self.values['FQE'] = self.naive_FQE

            # DM
            self.values['MA-FQE'] = np.mean(np.mean(V_mm, axis=1))
            self.values['BA-FQE'] = np.mean(np.mean(V_bm, axis=1)) 
            self.values['MM-FQE'] = np.mean(np.median(V_mm, axis=1)) 
            self.values['BM-FQE'] = np.mean(np.median(V_bm, axis=1)) 
            # IS
            self.est_IS()
            self.values['MM-IS'] = np.median(self.IS_it)
            self.values['MA-IS'] = np.mean(self.IS_it)
            self.est_IS(trajs_splits=self.boot_trajs_splits)
            self.values['BM-IS'] = np.median(self.IS_it)
            self.values['BA-IS'] = np.mean(self.IS_it)

    def est_Q2(self, verbose=1, test_freq=10, run_internal_Q_MM=True, **FQE_paras):
        """ 
        Q_func(self, S, A = None)
        self.ohio_eval.init_state
        
        TBD: for this func, I have changed sample spliting to mm-type spliting
        """        
        self.values = {}
        self.runtimes = {}

        ######### Estimate Q for every split  #########
        start_t = time()
        self.init_V_splits = {}
        for k in range(self.L):
            ##################
            value_func = FQE_module.FQE(
                policy = self.pi,                  # policy to be evaluated
                # num_actions = self.num_A, 
                # gamma = self.gamma, 
                init_states = self.init_S,         # used for evaluation
                gpu_number = self.gpu_number, **FQE_paras)
            train_traj = self.trajs_splits[k] #[self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            value_func.train(
                train_traj, verbose = verbose, 
                #  test_freq = test_freq
            )
            ###########################
            init_V = value_func.init_state_value(init_states = self.init_S)
            self.init_V_splits[k] = init_V
            self.raw_Qs[k] = np.mean(init_V)
            self.value_funcs.append(value_func)       
        run_t = time() - start_t

        # eta
        self.raw_Q = np.median(self.raw_Qs)
        self.values['MM-FQE (eta)'] = self.raw_Q    # median over the final "integrated" value estimates
        self.runtimes['MM-FQE (eta)'] = run_t
        # because tp is deterministic, so Q = V 
        V_mm = np.median(np.stack([self.init_V_splits[k] for k in self.init_V_splits], 1), axis = 1)
        self.values['MM-FQE (Q/V)'] = np.mean(V_mm) # because tp is deterministic, so Q = V 
        self.runtimes['MM-FQE (Q/V)'] = run_t

        # baseline I:
        start_t = time()
        self.est_naive_FQE(verbose=verbose, train_freq=test_freq, **FQE_paras)
        run_t = time() - start_t
        self.values['FQE'] = self.naive_FQE
        self.runtimes['FQE'] = run_t

        # Internal:
        self.mm_internal_Q = FQE_mm_internal.FQE(policy = self.pi, gpu_number=self.gpu_number, K=len(self.trajs_splits), **FQE_paras)
        start_t = time()
        self.mm_internal_Q.train(trajs_splits=self.trajs_splits, train_freq=test_freq, verbose=1, 
                                 nn_verbose=0, validation_freq=1)
        run_t = time() - start_t
        init_V = self.mm_internal_Q.init_state_value(init_states = self.init_S)
        self.values['MM-FQE (Internal)'] = np.mean(init_V)
        self.runtimes['MM-FQE (Internal)'] = run_t

    def est_Q(self, verbose=1, test_freq=10, run_internal_Q_MM=True, **FQE_paras):
        """ Q_func(self, S, A = None)
        self.ohio_eval.init_state
        
        TBD: for this func, I have changed sample spliting to mm-type spliting
        """        
        self.values = {}
        ######### Estimate Q for every split  #########
        self.init_V_splits = {}
        for k in range(self.L):
            ##################
            value_func = FQE_module.FQE(
                policy = self.pi,                  # policy to be evaluated
                # num_actions = self.num_A, 
                # gamma = self.gamma, 
                init_states = self.init_S,         # used for evaluation
                gpu_number = self.gpu_number, **FQE_paras)
            train_traj = self.trajs_splits[k] #[self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            value_func.train(
                train_traj, verbose = verbose, 
                #  test_freq = test_freq
            )
            ###########################
            init_V = value_func.init_state_value(init_states = self.init_S)
            self.init_V_splits[k] = init_V
            self.raw_Qs[k] = np.mean(init_V)
            self.value_funcs.append(value_func)       
            ### the behav value is significant affected by the initial
            ## stationary?            
            
#             disc_w_init = mean([sum([SA[2] * self.gamma ** t for t, SA in enumerate(traj)]) for traj in train_traj])
#             S, A, R, SS = [np.array([item[i] for traj in train_traj for item in traj]) for i in range(4)]
#             disc_w_all = np.mean(R / (1 - self.gamma))
            
#             S, A, R, SS = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
#             Q_S = self.value_funcs[k].Q_func(S, A)
#             Q_SS = self.value_funcs[k].Q_func(SS, self.pi.get_A(SS))
#             sampled_Qs = self.value_funcs[k].Q_func(self.init_S, self.pi.get_A(self.init_S))
#             self.Q_values[k] = {"Q_S" : Q_S.copy(), "Q_SS" : Q_SS.copy(), "sampled_Qs" : sampled_Qs.copy()}
#             if self.verbose:
#                 printR("behav value: disc_w_init = {:.2f} and disc_w_all = {:.2f}".format(disc_w_init, disc_w_all))
#                 printR("OPE init_Q: mean = {:.2f} and std = {:.2f}".format(np.mean(init_V)
#                                                                                      , np.std(init_V) / np.sqrt(len(self.init_S))))
#                 printG("<------------- FQE for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
                
        ######################################################
        """ ???
        old: FQE: average of FQE over replications
        new: just one run
        """
        # baseline I (naive):
        self.est_naive_FQE(verbose=verbose, train_freq=test_freq, **FQE_paras)
        self.values['FQE'] = self.naive_FQE
        # baseline II (truncated-mean): 
        # self.est_naive_FQE(verbose=verbose, train_freq=test_freq, truncate=0.2, **FQE_paras)
        # self.values['FQE-T'] = self.naive_FQE
        # eta
        self.raw_Q = np.median(self.raw_Qs)
        self.values['MM-FQE (eta)'] = self.raw_Q    # median over the final "integrated" value estimates
        # # because tp is deterministic, so Q = V 
        V_mm = np.median(np.stack([self.init_V_splits[k] for k in self.init_V_splits], 1), axis = 1)
        self.values['MM-FQE (Q/V)'] = np.mean(V_mm) # because tp is deterministic, so Q = V 
        V_mm = np.mean(np.stack([self.init_V_splits[k] for k in self.init_V_splits], 1), axis = 1)
        self.values['Mean-FQE (Q/V)'] = np.mean(V_mm) # because tp is deterministic, so Q = V 
        
        # IS
        self.est_IS()
        self.values['MM-IS'] = np.median(self.IS_it)
        # baseline IS-I: 
        self.values['Mean-IS'] = np.mean(self.IS_it)
        # baseline IS-II (truncated across folds): 
        self.values['Trunc-Mean-IS'] = truncate_mean(self.IS_it, alpha=0.01)
        # # baseline IS-III (standard truncated): 
        # self.est_trunc_IS()
        # self.values['Trunc-Mean-IS-1'] = self.trunc_IS_it

        if run_internal_Q_MM:
            self.run_mm_internal(verbose = verbose, train_freq = test_freq, FQE_paras = FQE_paras)

        #self.estimate_Geo_Median_Q()
        #self.GeoMedian_Q_values = self.GeoMedian_Q.init_state_value(init_states = self.init_S)
        #self.GeoMM_FQE_Q = np.mean(self.GeoMedian_Q_values)
        
#     def estimate_Geo_Median_Q(self):
#         replay_buffer = sampler.SimpleReplayBuffer(trajs = self.trajs)
#         self.GeoMedian_Q = GeoMedian_functional.GeoMedian(policy = self.pi, replay_buffer = replay_buffer, Qs = self.value_funcs
#                          , S_dim = self.S_dims
#                          , gpu_number = 0
#                          , hiddens= 256, nn_verbose = 0, es_patience=5, lr=1e-4
#         )
#         self.GeoMedian_Q.fit(batch_size = 32, max_iter=500, print_freq = 20, tolerance = 20)

    def est_IS(self, trajs_splits=None, h_dims=32, max_iter=100, batch_size=32, lr=0.0002, print_freq=20, tolerance=5, rep_loss=3): 
        if trajs_splits is None:
            trajs_splits = self.trajs_splits
        for k in range(self.L):
            curr_time = now()
            ###
            omega_func = omega_SA.VisitationRatioModel_init_SA(
                replay_buffer=sampler.SimpleReplayBuffer(trajs=trajs_splits[k]),
                target_policy=self.pi, A_range=self.A_range, h_dims=h_dims, lr=lr, gpu_number=self.gpu_number, sepe_A=self.sepe_A)

            omega_func.fit(batch_size=batch_size, gamma=self.gamma, max_iter=max_iter, 
                           print_freq=print_freq, tolerance=tolerance, rep_loss=rep_loss)
            if self.verbose:
                printG("<------------- omega estimation for fold {} DONE! Time cost = {:.1f} minutes ------------->".format(k, (now() - curr_time) / 60))
            ### 
            S, A, R, _ = [np.array([item[i] for traj in [self.trajs[j] for j in self.split_ind[k]["test_ind"]] for item in traj]) for i in range(4)]
            #########
            omega = omega_func.model.predict_4_VE(inputs = tf.concat([S, A[:,np.newaxis]], axis=-1)) #  (NT,)
            omega = np.squeeze(omega)
            ISs = (omega * R) / (1 - self.gamma) / np.mean(omega)
            if self.verbose:
                printR("IS for fold {} = {:.2f}".format(k, np.mean(ISs)))
            self.IS_it[k] = np.mean(ISs)


    def est_trunc_IS(self, h_dims=32, max_iter=100, batch_size=32, lr=0.0002, print_freq=20, tolerance=5, rep_loss=3): 
        curr_time = now()
        omega_func = omega_SA.VisitationRatioModel_init_SA(
            replay_buffer=sampler.SimpleReplayBuffer(trajs=self.complete_trajs),
            target_policy=self.pi, A_range=self.A_range, h_dims=h_dims, lr=lr, gpu_number=self.gpu_number, sepe_A=self.sepe_A)

        omega_func.fit(batch_size=batch_size, gamma=self.gamma, max_iter=max_iter, 
                        print_freq=print_freq, tolerance=tolerance, rep_loss=rep_loss)
        if self.verbose:
            printG("<------------- omega estimation DONE! Time cost = {:.1f} minutes ------------->".format((now() - curr_time) / 60))
        ### 
        S, A, R, _ = [np.array([item[i] for traj in [self.trajs[j] for j in range(self.N)] for item in traj]) for i in range(4)]
        #########
        omega = omega_func.model.predict_4_VE(inputs = tf.concat([S, A[:,np.newaxis]], axis=-1)) #  (NT,)
        omega = np.squeeze(omega)
        ISs = (omega * R) / (1 - self.gamma) / np.mean(omega)
        self.trunc_IS_it = truncate_mean(ISs, alpha=0.1)

    def est_naive_FQE(self, verbose=1, train_freq=10, truncate=0.0, **FQE_paras):
        value_func = FQE_module.FQE(policy = self.pi, # policy to be evaluated
                 #, num_actions = self.num_A, gamma = self.gamma
                init_states = self.init_S, # used for evaluation
                gpu_number = self.gpu_number, **FQE_paras)
        train_traj = self.trajs
        value_func.train(train_traj, verbose = verbose, train_freq = train_freq)
        ###########################
        init_V = value_func.init_state_value(init_states = self.init_S)
        self.naive_FQE = np.mean(init_V)
        if truncate > 0.0:
            self.naive_FQE = truncate_mean(init_V, alpha=0.1)
        
    # def esl_FQE(self, verbose = 1, train_freq = 10, **FQE_paras):
    #     value_func = FQE_module.ESL_FQE(
    #         policy = self.pi, # policy to be evaluated
    #         # num_actions = self.num_A, gamma = self.gamma
    #         init_states = self.init_S, # used for evaluation
    #         gpu_number = self.gpu_number, 
    #         **FQE_paras
    #     )
    #     train_traj = self.trajs
    #     value_func.train(
    #         train_traj, verbose = verbose, 
    #         train_freq = train_freq
    #     )
    #     init_V = value_func.init_state_value(init_states = self.init_S)
    #     self.esl_FQE = np.mean(init_V)
        
    def run_mm_internal(self, verbose = 1, train_freq = 10, **FQE_paras):
        self.mm_internal_Q = FQE_mm_internal.FQE(policy = self.pi, gpu_number = self.gpu_number, K = len(self.trajs_splits), **FQE_paras)
        self.mm_internal_Q.train(trajs_splits = self.trajs_splits, train_freq = train_freq, verbose = verbose, 
                                 nn_verbose = 0, validation_freq = 1)
        init_V = self.mm_internal_Q.init_state_value(init_states = self.init_S)
        self.values['MM-FQE (Internal)'] = np.mean(init_V)

