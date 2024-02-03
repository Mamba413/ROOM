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
L2_PENALTY = 20.0

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
class FQI(object):
    """ only works for binary action for now.
    """
    @autoargs()
    def __init__(self, num_actions=5, init_states = None, estimator = 'poly', 
                 gamma=0.99, trajs = None
                 , gpu_number = None, hiddens = None
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, tau = None
                 , validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001):
        ### === network ===
#         self.num_actions = num_actions
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
        if self.estimator in ['rkhs', 'rbf']:
            self._compute_medians(states, actions)
        if self.estimator == 'rkhs':
            self.model = GridSearchCV(
            KernelRidge(kernel="rbf", gamma = self.kernel_gamma),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]},
            )
        if self.estimator == 'poly':
            self.featurize = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True, include_bias=True)#(centered==False)
            states = self.featurize.fit_transform(states)
            next_states = self.featurize.fit_transform(next_states)
            # self.model = LinearRegression(fit_intercept=False)
            self.model = Ridge(fit_intercept=False, alpha=L2_PENALTY, random_state=0)
        self.models = [copy(self.model) for _ in range(self.num_actions)]
        #####
        old_targets = rewards / (1 - self.gamma)
        targets_diff_actions = [old_targets[actions.astype(int) == i] for i in range(self.num_actions)]
        states_diff_actions = [states[actions.astype(int) == i] for i in range(self.num_actions)]
        #####
        """
        form the target value (use observed i.o. the max) -> fit 
        """
        # FQE

        #print('old_targets', max(old_targets), mean(np.abs(old_targets)))
        # https://keras.rstudio.com/reference/fit.html
        self.states_diff_actions = states_diff_actions
        self.old_targets = old_targets
                
        for i in range(self.num_actions):
            self.models[i].fit(self.states_diff_actions[i], targets_diff_actions[i])
            if self.estimator == 'rkhs':
                self.best_alphas.append(self.models[i].best_params_['alpha'])
        return states, actions, rewards, next_states, old_targets
    
    def train(self, trajs, train_freq = 100, verbose=0, nn_verbose = 0, validation_freq = 1
             , path = None, save_freq = 10, es = True):
        
        states, actions, rewards, next_states, old_targets = self._init_train(trajs)
        self.nn_verbose = nn_verbose
        ###############################################################
        for iteration in range(self.max_iter):
            """ learn the FQI formula """
            self.iteration = iteration
            
            q_next_states = [self.models[i].predict(next_states) for i in range(self.num_actions)]
            q_next_states = np.stack(q_next_states, axis = 1)
            # q_next_states = np.clip(q_next_states, -10000, 10000)
            
            targets = rewards + self.gamma * np.max(q_next_states, 1)
#             a = [max(q1_next_states), mean(np.abs(q1_next_states))
#                   , max(rewards), mean(np.abs(rewards))
#                   , max(targets), mean(np.abs(targets))]
#             print(iteration
#                   , np.round(a, 2)
#                  )


            targets_diff_actions = [targets[actions.astype(int) == i] for i in range(self.num_actions)]
            
            ##
            if self.estimator == 'rkhs':
                if iteration < 5:
                    for i in range(self.num_actions):
                        self.models[i].fit(self.states_diff_actions[i], targets_diff_actions[i])
                        self.best_alphas.append(self.models[i].best_params_['alpha'])
                else:
                    if iteration == 5:
                        self.best_alpha = sp.stats.mode(self.best_alphas)[0][0]
                        self.model = KernelRidge(alpha = self.best_alpha, kernel = 'rbf', gamma = self.kernel_gamma)
                        self.models = [copy(self.model) for _ in range(self.num_actions)]
                    for i in range(self.num_actions):
                        self.models[i].fit(self.states_diff_actions[i], targets_diff_actions[i])
            else:
                for i in range(self.num_actions):
                    self.models[i].fit(self.states_diff_actions[i], targets_diff_actions[i])


            target_diff = change_rate(old_targets, targets)
            self.target_diffs.append(target_diff)

            # print(iteration, target_diff)
            
            if verbose >= 1 and iteration % train_freq == 0 and self.gpu_number % 8 == 0:
                print('----- FQI (training) iteration: {}, target_diff = {:.3f}'.format(iteration, target_diff, '-----'))
            if target_diff < self.eps:
                break

            old_targets = targets.copy()

            ################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################            
    """ self.model.predict
    Q values? or their action probabilities???????????

    """
    def Q_func(self, states, actions = None):

        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if self.estimator == 'poly':
            states = self.featurize.fit_transform(states)
        if actions is not None:
            pass
            #return np.squeeze(select_each_row(self.model(states), actions.astype(int)))
        else:
            Q = [self.models[i].predict(states) for i in range(self.num_actions)]
            Q = np.stack(Q, axis = 1)
            return Q

    def sample_A(self, states):
        if self.tau is not None:
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
            Qs = self.Q_func(states)
            logit = np.exp(Qs / self.tau)
            probs = logit / np.sum(logit, 1)[:, np.newaxis]
            As = [np.random.choice(self.num_actions, size = 1, p = aa)[0] for aa in probs]
            return As
        else:
            return self.get_A(states)

    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.amax(self.Q_func(states), axis=1)
    
    
    def A_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if actions is not None:
            return np.squeeze(self.Q_func(states, actions.astype(int))) - self.V_func(states)
        else:
            return transpose(transpose(self.Q_func(states)) - self.V_func(states)) # transpose so that to subtract V from Q in batch.     
        
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
    
    def get_A(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.argmax(self.Q_func(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_actions
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        probs = np.zeros((len(states), self.num_actions))
        probs[range(len(states)), optimal_actions] = 1
        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
        


    
def change_rate(old_targets, new_targets):
    diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)


class MM_Q_policy(object):
    """ compute the median of the Qs of the policys, and make the median into a policy
    """
    def __init__(self, policys, num_actions=5, init_states = None, pessimistic = False, estimator = 'poly', 
                 gamma=0.99, gpu_number = 0
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, max_epoch=200, tau = None
                 , validation_split=0.2, es_patience=20
                 , max_iter=100, eps=0.001, pess_quantile=0.4):
        self.policys = policys
        self.pessimistic = pessimistic
        ### === network ===
        self.num_actions = num_actions
        self.estimator = estimator
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.init_states = init_states
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        self.gpu_number = gpu_number
        
        self.tau = tau
        self.pess_quantile = pess_quantile
    
    
    def Q_func(self, states, actions = None):

        Qs = []
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
#         states = (states - self.mean_S) / self.std_S
        if self.estimator == 'poly':
            self.featurize = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True, include_bias=True)#(centered==False)
            states = self.featurize.fit_transform(states)
        for policy in self.policys: 
            if actions is not None:
                pass
#                 Q = policy.model(states)
#                 Q = np.squeeze(select_each_row(Q, actions.astype(int)))
#                 Qs.append(Q) 
            else:
                Q = [policy.models[i].predict(states) for i in range(self.num_actions)]
                Q = np.stack(Q, axis = 1)
                #Q = policy.model(states)
                Qs.append(Q) 

        #Q = np.median(np.stack([Q for Q in Qs], 2), axis = 2)
        Qs = np.stack([Q for Q in Qs], 2)
        if self.pessimistic:
            # Q = np.quantile(Qs, q=self.pess_quantile, method='lower', axis = 2)
            Q = np.quantile(Qs, q=self.pess_quantile, axis = 2)
        else:
            Q = np.median(Qs, axis = 2)

        return Q


    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.amax(self.Q_func(states), axis=1)
    
    
    def A_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if actions is not None:
            return np.squeeze(self.Q_func(states, actions.astype(int))) - self.V_func(states)
        else:
            return transpose(transpose(self.Q_func(states)) - self.V_func(states)) # transpose so that to subtract V from Q in batch.     
        
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
    
    def get_A(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.argmax(self.Q_func(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_actions
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        probs = np.zeros((len(states), self.num_actions))
        probs[range(len(states)), optimal_actions] = 1
        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
        
    def sample_A(self, states):
        if self.tau is not None:
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
            Qs = self.Q_func(states)
            logit = np.exp(Qs / self.tau)
            probs = logit / np.sum(logit, 1)[:, np.newaxis]
            As = [np.random.choice(self.num_actions, size = 1, p = aa)[0] for aa in probs]
            return As
        else:
            return self.get_A(states)


class PB_Q_policy(object):
    """ compute the median of the Qs of the policys, and make the median into a policy
    """
    def __init__(self, policys, num_actions=5, init_states = None, pessimistic = True, estimator = 'poly', 
                 gamma=0.99, gpu_number = 0, 
                 lr=5e-4, decay_steps=100000, 
                 batch_size=64, max_epoch=200, tau = None, 
                 validation_split=0.2, es_patience=20, 
                 max_iter=100, eps=0.001, pess_quantile=0.4, 
                 pess_scale_std=1.0):
        self.policys = policys
        self.pessimistic = pessimistic
        ### === network ===
        self.num_actions = num_actions
        self.estimator = estimator
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.init_states = init_states
        ### === optimization ===
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter
        
        self.target_diffs = []
        self.values = []
        self.gpu_number = gpu_number
        
        self.tau = tau
        self.pess_quantile = pess_quantile
        self.pess_scale_std = pess_scale_std
    
    
    def Q_func(self, states, actions = None):
        Qs = []
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if self.estimator == 'poly':
            self.featurize = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True, include_bias=True)#(centered==False)
            states = self.featurize.fit_transform(states)
        for policy in self.policys: 
            if actions is not None:
                pass
            else:
                Q = [policy.models[i].predict(states) for i in range(self.num_actions)]
                Q = np.stack(Q, axis = 1)
                Qs.append(Q) 

        Qs = np.stack([Q for Q in Qs], 2)
        Q_mean = np.mean(Qs, axis = 2)
        Q_std = np.std(Qs, axis = 2)
        Q = Q_mean - self.pess_scale_std * Q_std

        return Q


    def V_func(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.amax(self.Q_func(states), axis=1)
    
    
    def A_func(self, states, actions = None):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if actions is not None:
            return np.squeeze(self.Q_func(states, actions.astype(int))) - self.V_func(states)
        else:
            return transpose(transpose(self.Q_func(states)) - self.V_func(states)) # transpose so that to subtract V from Q in batch.     
        
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
    
    def get_A(self, states):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        return np.argmax(self.Q_func(states), axis = 1)

    def get_A_prob(self, states, actions = None, multi_dim = False, to_numpy= True):
        """ probability for one action / prob matrix """
        if multi_dim and len(states) > 2:
            pre_dims = list(states.shape)
            pre_dims[-1] = self.num_actions
            states = states.reshape(-1, self.S_dims)
        
        optimal_actions = self.get_A(states)
        probs = np.zeros((len(states), self.num_actions))
        probs[range(len(states)), optimal_actions] = 1
        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
        
    def sample_A(self, states):
        if self.tau is not None:
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
            Qs = self.Q_func(states)
            logit = np.exp(Qs / self.tau)
            probs = logit / np.sum(logit, 1)[:, np.newaxis]
            As = [np.random.choice(self.num_actions, size = 1, p = aa)[0] for aa in probs]
            return As
        else:
            return self.get_A(states)


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