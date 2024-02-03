from _util import *
import _RL.FQE as FQE_module
import _RL.FQI as FQI
reload(FQE_module)
reload(FQI)
import _RL.sampler as sampler
# import GeoMedian_functional
# reload(GeoMedian_functional)
######################################################################################################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     from _density import omega_SA, omega_SASA
#     reload(omega_SA)
#     reload(omega_SASA)
#     import tensorflow as tf

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# tf.keras.backend.set_floatx('float64')
tf.keras.backend.set_floatx('float32')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

################################################################################################################################################################################################################################################################################################################################################################################################################################################

class MMOPO():
    """ 
    """
    @autoargs()
    def __init__(self, trajs, eval_N = 1000, gpu_number = 0, verbose = 0, seed = 0 
                 , L = 2, incomplete_ratio = 20, sepe_A = 0, A_range = [0, 1, 2, 3, 4]
                 , gamma = .95, pess_quantile=0.4):
        #self.trajs = trajs # data: T transition tuple for N trajectories
        self.S, self.A, self.R, self.SS = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        self.N, self.T, self.L = len(trajs), len(trajs[0]), L
        self.S_dims = len(np.atleast_1d(trajs[0][0][0]))
        self.A_range = arr(A_range).astype(np.float64) #set(self.A)
        self.num_A = len(self.A_range)
        self.split_ind = sample_split(self.L, self.N) # split_ind[k] = {"train_ind" : i, "test_ind" : j}
        ###
        self.value_funcs = []
        self.omegas = []
        self.omegas_values = []
        self.omegas_star = []
        self.omegas_star_values = []
        self.Q_values = {}
        self.raw_Qs = zeros(self.L)
        self.trajs_splits = [[self.trajs[i] for i in self.split_ind[k]["test_ind"]]  for k in range(self.L)]
        self.policies = {}
        self.policies_over_splits = []
        self.pess_quantile = pess_quantile

    ############################################################################################################################
    ###########################################  The three main components #####################################################
    ############################################################################################################################
    
    def learn_policies(self, verbose = 1, test_freq = 10, parallel = False, run_internal_Q_MM = True
                       , max_iter = 1000, batch_size = 256, eps = 0.00
                       , **FQE_paras
                      ):
        """ Q_func(self, S, A = None)
        self.ohio_eval.init_state
        
        TBD: for this func, I have changed sample spliting to mm-type spliting
        """        
        #########
        ###
        FQI_paras = {"hiddens" : [32, 32], "gamma" : self.gamma, 'num_actions' : self.num_A, 'batch_size' : batch_size
                    , 'eps' : eps}
        ###
        import _RL.FQI as FQI
        reload(FQI)
        for k in range(self.L):
            #curr_time = now()
            ##################
            train_traj = self.trajs_splits[k] #[self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            ########################################################################

            pi1 = FQI.FQI(max_iter = max_iter, gpu_number = self.gpu_number, **FQI_paras)
            pi1.train(trajs = train_traj, train_freq = test_freq, verbose = verbose, nn_verbose = 0, validation_freq = 1
                         , path = None, save_freq = 10
                         , es = False)
            self.policies_over_splits.append(pi1)
#             if self.seed % 10 == 0:
#                 print("FQI training for split", k + 1, 'is DONE!')

        self.run_mm_final()
        ### baseline
        self.run_naive_FQI(verbose = verbose, test_freq = test_freq, max_iter = max_iter, FQI_paras = FQI_paras)
        
        if run_internal_Q_MM:
            self.run_mm_internal(max_iter = max_iter, FQI_paras = FQI_paras, test_freq = test_freq, verbose = verbose)

    def run_mm_final(self):
        ### compute the median of the Qs of the policys, and make the median into a policy
        self.mm_final_Q_policy = FQI.MM_Q_policy(policys = self.policies_over_splits, 
                                                 num_actions = self.num_A, init_states = None, 
                                                 gamma = self.gamma, gpu_number = self.gpu_number, 
                                                 pess_quantile=self.pess_quantile)
        self.policies['mm_final'] = self.mm_final_Q_policy
        #########

        ### compute the median of the Qs of the policys, and make the median into a policy
        self.mm_final_Q_pessimistic_policy = FQI.MM_Q_policy(policys = self.policies_over_splits, 
                                                             num_actions = self.num_A, init_states = None, 
                                                             pessimistic = True, gamma = self.gamma, gpu_number = self.gpu_number, 
                                                             pess_quantile=self.pess_quantile)
        self.policies['mm_final_pessimistic'] = self.mm_final_Q_pessimistic_policy
        
    def run_mm_internal(self, max_iter = 1000, FQI_paras = None, test_freq = None, verbose = None):
        ### Internal MM
        import _RL.FQI_mm_internal as FQI_mm_internal
        reload(FQI_mm_internal)

        self.mm_internal_Q_policy = FQI_mm_internal.FQI(max_iter = max_iter, gpu_number = self.gpu_number, 
                                                        K = len(self.trajs_splits), pessimistic=False,  **FQI_paras)
        self.mm_internal_Q_policy.train(trajs_splits = self.trajs_splits, train_freq = test_freq, verbose = verbose, nn_verbose = 0, validation_freq = 1
                     , path = None, save_freq = 10
                                       , es = False)
        self.policies['mm_internal'] = self.mm_internal_Q_policy
        
        
        self.mm_internal_Q_pessimistic_policy = FQI_mm_internal.FQI(max_iter = max_iter, gpu_number = self.gpu_number, 
                                                                    K = len(self.trajs_splits), pessimistic =True, 
                                                                    pess_quantile=self.pess_quantile,  **FQI_paras)
        self.mm_internal_Q_pessimistic_policy.train(trajs_splits = self.trajs_splits, train_freq = test_freq, 
                                                    verbose = verbose, nn_verbose = 0, validation_freq = 1, 
                                                    path = None, save_freq = 10, es = False)
        self.policies['mm_internal_pessimistic'] = self.mm_internal_Q_pessimistic_policy


            
    def run_naive_FQI(self, verbose = 1, test_freq = 10, max_iter = 1000, FQI_paras = None
                      #, **FQE_paras
                     ):
        
        
        import _RL.FQI as FQI
        reload(FQI)
        self.naive_FQI_policy = FQI.FQI(max_iter = max_iter, gpu_number = self.gpu_number, **FQI_paras)
        self.naive_FQI_policy.train(trajs = self.trajs, train_freq = test_freq, verbose = verbose, nn_verbose = 0, validation_freq = 1
                     , path = None, save_freq = 10
                                   , es = False)
        self.policies['naive'] = self.naive_FQI_policy
    ############################################################################################################################
#     def is_diff(self, old, new):
#         old = np.concatenate(old)
#         new = np.concatenate(new)
#         diff = np.mean(new) - np.mean(old)
#         std = np.std(arr(new) - arr(old)) / np.sqrt(len(old))
#         z = np.abs(diff) / std
#         alpha = 0.05
#         if z > sp.stats.norm.ppf((1 - alpha / 2)):
#             return True
#         else:
#             return False
    ############################################################################################################################
    ###########################################  Evaluation #####################################################
    ############################################################################################################################
#     def learn_one_policy(self, gpu_number, train_traj, FQI_paras, verbose = 1, test_freq = 10):
#         reload(FQI)
#         ########################################################################
#         pi1 = FQI.FQI(max_iter = 30, gpu_number = gpu_number, **FQI_paras)
#         pi1.train(trajs = train_traj, train_freq = test_freq, verbose = verbose, nn_verbose = 0, validation_freq = 1
#                              , path = None, save_freq = 10)
#         return pi1
    
    
#         import ray
#can't pickle weakref objects
#         if parallel:
#             ###########
#             try:
#                 ray.shutdown()
#             except:
#                 pass
#             @ray.remote(num_gpus=1)
#             def one_seed1(gpu_number, train_traj, FQI_paras):
#                 pi = self.learn_one_policy(gpu_number, train_traj, FQI_paras, verbose = verbose, test_freq = test_freq)
#                 return pi
            
#             ###########

#             ray.init()
#             futures = [one_seed1.remote(gpu_number, train_traj, FQI_paras) for gpu_number, train_traj in 
#                        zip(range(self.L), trajs_splits)]
#             self.policies_over_splits = ray.get(futures)
#             # for j in range(n_gpu):
#             #     rec.update(V_true, are_details = res[j])
#             # rec.analyze()
#             # rec.save("res/" + setting)
#             ray.shutdown()
#         else:
