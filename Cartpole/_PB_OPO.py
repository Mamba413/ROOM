import os
import _RL.sampler as sampler
from _util import *
import _RL.FQE as FQE_module
import _RL.FQI as FQI
reload(FQE_module)
reload(FQI)
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

################################################################################################################################################################################################################################################################################################################################################################################################################################################


class PBOPO():
    """ 
    """
    @autoargs()
    def __init__(self, trajs, eval_N=1000, gpu_number=0, verbose=0, seed=0, L=2, incomplete_ratio=20, sepe_A=0, A_range=[0, 1, 2, 3, 4], gamma=.9, pess_quantile=0.4):
        #self.trajs = trajs # data: T transition tuple for N trajectories
        self.S, self.A, self.R, self.SS = [
            np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        self.N, self.T, self.L = len(trajs), len(trajs[0]), L
        self.S_dims = len(np.atleast_1d(trajs[0][0][0]))
        self.A_range = arr(A_range).astype(np.float64)  # set(self.A)
        self.num_A = len(self.A_range)
        self.pess_quantile = pess_quantile
        # split_ind[k] = {"train_ind" : i, "test_ind" : j}
        self.split_ind = bootstrapping_sampling(self.L, self.N)
        ###
        self.value_funcs = []
        self.omegas = []
        self.omegas_values = []
        self.omegas_star = []
        self.omegas_star_values = []
        self.Q_values = {}
        self.raw_Qs = zeros(self.L)
        self.trajs_splits = [
            [self.trajs[i] for i in self.split_ind[k]["test_ind"]] for k in range(self.L)]
        self.policies = {}
        self.policies_over_splits = []

    ############################################################################################################################
    ###########################################  The three main components #####################################################
    ############################################################################################################################

    def learn_policies(self, verbose=1, test_freq=10, parallel=False, run_internal_Q_PB=True, max_iter=1000, batch_size=256, eps=0.005, **FQE_paras
                       ):
        """ Q_func(self, S, A = None)
        self.ohio_eval.init_state
        
        TBD: for this func, I have changed sample spliting to mm-type spliting
        """
        #########
        ###
        FQI_paras = {"hiddens": [32, 32], "gamma": self.gamma,
                     'num_actions': self.num_A, 'batch_size': batch_size, 'eps': eps,}
        ###
        import _RL.FQI as FQI
        reload(FQI)
        for k in range(self.L):
            #curr_time = now()
            ##################
            # [self.trajs[i] for i in self.split_ind[k]["test_ind"]]
            train_traj = self.trajs_splits[k]
            ########################################################################

            pi1 = FQI.FQI(max_iter=max_iter,
                          gpu_number=self.gpu_number, **FQI_paras)
            pi1.train(trajs=train_traj, train_freq=test_freq, verbose=verbose,
                      nn_verbose=0, validation_freq=1, path=None, save_freq=10, es=False)
            self.policies_over_splits.append(pi1)
#             if self.seed % 10 == 0:
#                 print("FQI training for split", k + 1, 'is DONE!')

        self.run_pb_final()

        if run_internal_Q_PB:
            self.run_pb_internal(
                max_iter=max_iter, FQI_paras=FQI_paras, test_freq=test_freq, verbose=verbose)

    def run_pb_final(self):
        self.mm_final_Q_policy = FQI.PB_Q_policy(policys=self.policies_over_splits, num_actions=self.num_A,
                                                 pessimistic=True, init_states=None,
                                                 gamma=self.gamma, gpu_number=self.gpu_number, 
                                                 pess_quantile=self.pess_quantile)
        self.policies['pb_final'] = self.mm_final_Q_policy

    def run_pb_internal(self, max_iter=1000, FQI_paras=None, test_freq=None, verbose=None):
        ### Internal MM
        import _RL.FQI_mm_internal as FQI_mm_internal
        reload(FQI_mm_internal)

        self.pb_internal_Q_pessimistic_policy = FQI_mm_internal.FQI(
            max_iter=max_iter, gpu_number=self.gpu_number, K=len(self.trajs_splits), pessimistic=True, pess_quantile=self.pess_quantile, **FQI_paras)
        self.pb_internal_Q_pessimistic_policy.train(
            train_freq=test_freq, verbose=verbose, nn_verbose=0, validation_freq=1, path=None, save_freq=10, es=False, 
            trajs_splits=self.trajs_splits, quantile_bound=False,   # key difference between PB-based and MM-based method
        )
        self.policies['pb_internal'] = self.pb_internal_Q_pessimistic_policy
