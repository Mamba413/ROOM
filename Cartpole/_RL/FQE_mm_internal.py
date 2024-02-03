### Adapt from ###
from _util import *
import numpy as np
import tensorflow as tf


import collections
import collections
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from copy import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.kernel_approximation import RBFSampler
# from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
# from joblib import Parallel, delayed
from bisect import bisect_right
import csv

POLY_DEGREE = 2
L2_PENALTY = 0.01

from _util import *

"""
batch RL
deterministic policies <- Q-based
model is the Q-network
"""

"""
if self.qmodel == "rbf":
    self.featurize = RBFSampler(gamma=rbf_bw, random_state=RBFSampler_random_state, n_components=rbf_dim)

"""
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class FQE(object):
    """ only works for binary action for now.
    """
    @autoargs()
    def __init__(self, policy = None
                 , num_actions=2, init_states = None, estimator = 'poly', K = 5, 
                 gamma=0.99, trajs = None
                 , gpu_number = None, hiddens = None
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, tau = None
                 , validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001, 
                 **FQE_paras
                 ):
#         ### === network ===
#         self.num_actions = num_actions
#         self.K = K
#         self.pessimistic = pessimistic
#         self.init_states = init_states
#         self.estimator = estimator
#         ### === optimization ===
#         self.batch_size = batch_size
#         self.max_epoch = max_epoch
#         self.validation_split = validation_split
#         # discount factor
#         self.gamma = gamma
#         self.eps = eps
#         self.max_iter = max_iter
#         self.gpu_number = gpu_number
        
        self.target_diffs = []
        self.values = []
        
#         self.tau = tau
        self.best_alphas = []

            
    def _compute_medians(self, states, actions):
        # to save memory, we do it iteratively
        self.S_dims = len(states[0])
        n = len(states)
        S, A = states, actions
#         dSS = np.repeat(S, n, axis=0) - np.tile(S, [n, 1])
#         median_S = np.median(np.abs(dSS), axis=0)        
#         self.median_S = median_S * self.S_dims
#         self.median_S = np.median(self.median_S)  # seems only one dim is allowed?
        ## Updated on 05/30/2022
        self.median_S = np.median(sk.metrics.pairwise_distances(S))
        self.kernel_gamma = 1 / (self.median_S ** 2) # gamma = sigma^{-2}, based on sk
        
    def _init_train(self, trajs):
        self.trajs = trajs
        states, actions, rewards, next_states = [np.array([item[i] for traj in trajs for item in traj]) for i in range(4)]
        _actions = self.policy.get_A(next_states)
        if self.estimator in ['rkhs', 'rbf']:
            self._compute_medians(states, actions)
        if self.estimator == 'rkhs':
            model = GridSearchCV(
            KernelRidge(kernel="rbf", gamma = self.kernel_gamma),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]},
            )
        if self.estimator == 'poly':
            self.featurize = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True, include_bias=True)#(centered==False)
            states = self.featurize.fit_transform(states)
            next_states = self.featurize.fit_transform(next_states)
            # model = LinearRegression(fit_intercept=False)
            model = Ridge(fit_intercept=False, alpha=L2_PENALTY, random_state=0)
        models = [copy(model) for _ in range(self.num_actions)]
        #####
        old_targets = rewards / (1 - self.gamma)
        targets_diff_actions = [old_targets[actions.astype(int) == i] for i in range(self.num_actions)]
        states_diff_actions = [states[actions.astype(int) == i] for i in range(self.num_actions)]
        
        #####
        """
        form the target value (use observed i.o. the max) -> fit 
        """

        #print('old_targets', max(old_targets), mean(np.abs(old_targets)))
        # https://keras.rstudio.com/reference/fit.html
        states_diff_actions = states_diff_actions
                
        for i in range(self.num_actions):
            models[i].fit(states_diff_actions[i], targets_diff_actions[i])
            if self.estimator == 'rkhs':
                self.best_alphas.append(models[i].best_params_['alpha'])
        return states, actions, rewards, next_states, old_targets, models, states_diff_actions, _actions
    
    def train(self, trajs_splits, train_freq = 100, verbose=0, nn_verbose = 0, validation_freq = 1
             , path = None, save_freq = 10, es = True):
        
        trajs_splits_details = []
        old_targets_splits = []
        self.models_splits = []
        self.states_diff_actions_splits = []
        self._actions_splits = []
        for k in range(self.K): 
            states, actions, rewards, next_states, old_targets, models, states_diff_actions, _actions = self._init_train(trajs_splits[k])
            trajs_splits_details.append([states, actions, rewards, next_states])
            old_targets_splits.append(old_targets)
            self.models_splits.append(models)
            self.states_diff_actions_splits.append(states_diff_actions)
            self._actions_splits.append(_actions)
        self.nn_verbose = nn_verbose
        
        ###############################################################
        
        for iteration in range(self.max_iter):
            """ learn the FQI formula """
            self.iteration = iteration
            self.target_diffs = []
            for k in range(self.K): 
                ## compute the targets, based on the models from the last iteration
                states, actions, rewards, next_states = trajs_splits_details[k]
#                 q_next_states = [self.models_splits[k][i].predict(next_states) for i in range(self.num_actions)]
#                 q_next_states = np.stack(q_next_states, axis = 1)
#                 q_next_states = np.clip(q_next_states, -10000, 10000)
                
                
                ## for these samples (belonging to the k-th split, use all models to compute the median)
                q_next_states_splits = [np.stack([self.models_splits[k][i].predict(next_states) for i in range(self.num_actions)], axis = 1) for k in range(self.K)]
                Q_this_batch_stack_over_splits = np.stack(q_next_states_splits, 2)
                Q_this_batch_stack_over_splits = np.clip(Q_this_batch_stack_over_splits, -10000, 10000)
                #np.quantile(np.arange(1, 11), 0.4, interpolation = 'lower') = 4
                #np.quantile(np.arange(1, 6), 0.4, interpolation = 'lower') = 2
#                 if self.pessimistic:
#                     q_next_states = np.quantile(Q_this_batch_stack_over_splits, q = 0.4, interpolation = 'lower'
#                                                 , axis = 2) # hard coded when L = 5
#                 else:
                q_next_states = np.median(Q_this_batch_stack_over_splits, axis = 2)


                #targets = rewards + self.gamma * np.max(q_next_states, 1)
                targets = rewards + self.gamma * q_next_states[range(len(self._actions_splits[k])), self._actions_splits[k]]
                targets_diff_actions = [targets[actions.astype(int) == i] for i in range(self.num_actions)]

                ##
                if self.estimator == 'rkhs':
                    if iteration < 5:
                        for i in range(self.num_actions):
                            self.models_splits[k][i].fit(self.states_diff_actions_splits[k][i], targets_diff_actions[i])
                            self.best_alphas.append(self.models_splits[k][i].best_params_['alpha'])
                    else:
                        # TODO
                        if iteration == 5:
                            self.best_alpha = sp.stats.mode(self.best_alphas)[0][0]
                            self.model = KernelRidge(alpha = self.best_alpha, kernel = 'rbf', gamma = self.kernel_gamma)
                            self.models = [copy(self.model) for _ in range(self.num_actions)]
                        for i in range(self.num_actions):
                            self.models[i].fit(self.states_diff_actions[i], targets_diff_actions[i])
                else:
                    for i in range(self.num_actions):
                        self.models_splits[k][i].fit(self.states_diff_actions_splits[k][i], targets_diff_actions[i])


                target_diff = change_rate(old_targets_splits[k], targets)
                self.target_diffs.append(target_diff)
                old_targets_splits[k] = targets.copy()
            

            # print(iteration, target_diff)
            
            if verbose >= 1 and iteration % train_freq == 0 and self.gpu_number % 8 == 0:
                print('----- FQI (training) iteration: {}, target_diff = {}'.format(iteration, np.round(self.target_diffs,3)
                                                                                    , '-----'))
            if max(self.target_diffs) < self.eps:
                break

            

            ################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################            
    """ self.model.predict
    Q values? or their action probabilities???????????

    """
    
    def Q_func(self, states, actions = None):

        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if self.estimator == 'poly':
            states = self.featurize.fit_transform(states)
        
        Qs = []
        for models in self.models_splits:
            Q = [models[i].predict(states) for i in range(self.num_actions)]
            Q = np.stack(Q, axis = 1)
            Qs.append(Q) 
                
        Qs = np.stack([Q for Q in Qs], 2)
#         if self.pessimistic:
#             Q = np.quantile(Qs, q = 0.4, interpolation = 'lower'
#                                         , axis = 2) # hard coded when L = 5
#         else:
        Q = np.median(Qs, axis = 2)
            
        if actions is not None:
            V = Q[range(len(actions)), actions]
            return V
        else:
            return Q


    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        _actions = self.policy.get_A(states) # deterministic; since FQI -> Q-based
        V = self.Q_func(states, _actions)
        return V
        
    def init_state_value(self, init_states = None, trajs = None, idx=0):
        """ TODO: Check definitions. 
        """
        if init_states is None:
            states = np.array([traj[idx][0] for traj in self.trajs])
        else:
            states = init_states
        return self.V_func(states) # len-n    
    """ NOTE: deterministic. for triply robust (off-policy learning)    
    """
            

    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)



#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class weight_policies():
    def __init__(self, pi1, pi2, w = 0.5):
        self.pi1 = pi1
        self.pi2 = pi2
        self.w = w # the weight of pi1
        
    def get_A(self, S):
        A_1 = self.pi1.sample_A(S)
        A_2 = self.pi2.sample_A(S)
        choice = np.random.binomial(n = 1, p = self.w, size = len(A_1))
        A = A_1 * choice + A_2 * (1 - choice)
        A = A.astype(np.int)
        return A

    def sample_A(self, S):
        return self.get_A(S)