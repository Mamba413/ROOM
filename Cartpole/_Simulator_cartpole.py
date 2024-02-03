"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from _util import *
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        # print(self.env_config)
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    try:
                        setattr(self, key,
                            type(getattr(self, key))(value))
                    except:
                        pass
            else:
                setattr(self, key, value)
                


class Simulator(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.

    ###
    std: noise to the four state variables
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    
    if DONE, then continue to the end of the horizon so that T is the same. -> but in that stedy case, how can we learn anything?
    """
    @autoargs()
    def __init__(self, env_config
#                  , 
#                  , std = 0.02
#                  , seed = 42
#                  , heavy_tailed = False, heavy_tail = 'levy', df = 1.2, alpha_levy = 1.1
#                  , reward_noise_std = 1
                ):
        self.e_max = 200
        assign_env_config(self, env_config) 
        ### 
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
#         self.std = np.repeat(0, 4)
        self.std = np.repeat(self.std, 4)
        self.e = 0
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        if self.heavy_tailed:
            if self.heavy_tail == 'levy':
                #rvs = levy_stable.rvs(alpha = self.alpha_levy, beta = 0, loc=0, scale=1, size = 10000, random_state=0)
                rvs = levy_stable.rvs(alpha = self.alpha_levy, beta = 0, loc = 0, scale=1, size = 10000, random_state=42)
            elif self.heavy_tail == 't':
                rvs = sp.stats.t.rvs(df = self.df, loc = 0, scale = 1, size = 10000, random_state = 42)
            """ 
            When heavy-tailed, the variance may not exist
            If we scale by varaince, for most entries, if might be very small, which is not acceptable?
            For a quick fix, I will clip and then compute the variance
            """
            clipped_rvs = np.clip(rvs, np.quantile(rvs, [0.01]), np.quantile(rvs, [0.99]))
            self.sample_std = np.std(clipped_rvs)

        seed = self.seed
        self.set_seed(seed = seed)
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def set_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def online_step(self, action):
        return self.step(action)[:3]
        
    def step(self, action):
        np.random.seed(self.seed)
        self.seed += 1
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        errors = np.random.randn(4) * self.std
        self.state = (x + errors[0], x_dot + errors[1], theta + errors[2], theta_dot + errors[3])
        # self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.e > self.e_max
        )
        self.e += 1
        if not done:
            """ reward """
            reward = self.cal_reward(x, theta) #1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            """ reward """
            reward = self.cal_reward(x, theta) #1.0
            
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            self.seed += 1
            reward = 0.0 + self.gen_noise(u = 0, noise_std = self.reward_noise_std, size = (1,), seed = self.seed)[0]
        return np.array(self.state), reward, done, {}

    def cal_reward(self, x, theta, u = 0, with_noise = True):
        # https://towardsdatascience.com/infinite-steps-cartpole-problem-with-variable-reward-7ad9a0dcf6d0
        # If x and θ represents cart position and pole angle respectively, we define the reward as:
        
        ## x, theta are too small -> minor difference to the reward
        reward = (1 - (x ** 2) / 11.52 - ((theta * 360 / math.pi / 2) ** 2) / 288) 
        #reward = (1 - (x ** 2) / 11.52 + (theta ** 2) / 288) 
        if with_noise:
            self.seed += 1
            reward_nose = self.gen_noise(u = 0, noise_std = self.reward_noise_std, size = reward.shape, seed = self.seed)
            reward = reward + reward_nose
        
        return reward
#         def angle_normalize(x):
#             return (((x+np.pi) % (2*np.pi)) - np.pi)
#         # angle_normalise((th)**2 +.1*thdot**2 + .001*(action**2))
#         costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        #return -costs
    
    def reset(self, T = None):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.e = 0
        return np.array(self.state)
    
    ############################################################################################################################################
    def gen_noise(self, u, noise_std, size, seed):
        if self.heavy_tailed:
            if self.heavy_tail == 'levy':
                #rvs = levy_stable.rvs(alpha = self.alpha_levy, beta = 0, loc=0, scale=1, size = 10000, random_state=0)
                rvs = levy_stable.rvs(alpha = self.alpha_levy, beta = 0, loc = u, scale=1, size = size, random_state=seed)
            elif self.heavy_tail == 't':
                rvs = sp.stats.t.rvs(df = self.df, loc = u, scale = 1, size = size, random_state = seed)

            """ 
            When heavy-tailed, the variance may not exist
            If we scale by varaince, for most entries, if might be very small, which is not acceptable?
            For a quick fix, I will clip and then compute the variance
            """
            # sample_std = np.clip(rvs, np.quantile(rvs, [0.01]), np.quantile(rvs, [0.99]))
            rvs = rvs / self.sample_std * noise_std #np.std(sample_std) * noise_std
            # rvs = rvs - np.mean(rvs) + u
        else:
            rvs = np.random.normal(size = size) * noise_std + u #randn(size)
        return rvs
    ##################################################################################################
    ##################################################################################################
    
    def reset_multiple(self, N, seed = 42):
        np.random.seed(seed)
        self.states = self.np_random.uniform(low=-0.05, high=0.05, size=(4, N))
        self.steps_beyond_done = np.repeat(None, N)
        self.dones = np.repeat(False, N)
        self.e = np.repeat(0, N)
        return np.array(self.states)
        
#         elif self.steps_beyond_done is None:
#             # Pole just fell!
#             self.steps_beyond_done = 0
#             """ reward """
#             reward = self.cal_reward(x, theta) #1.0
#         else:
#             if self.steps_beyond_done == 0:
#                 logger.warn(
#                     "You are calling 'step()' even though this "
#                     "environment has already returned done = True. You "
#                     "should always call 'reset()' once you receive 'done = "
#                     "True' -- any further steps are undefined behavior."
#                 )
#             self.steps_beyond_done += 1
#             reward = 0.0


#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None
############################################################################################################################################
############################################### For Batch-type sampling and evaluation ####################################################################
############################################################################################################################################

    def evaluate_policy(self, pi, gamma, init_states = None, N = 1000, T = 500, seed = 42, print_out = False):
        """ sample traj to analyze policy pi, with a given gamma
        init_states: whether to use the specified init states, or just sample
        """
        rep = N
        rewards = self.simu_trajs_parallel(pi, N = N, T = T, 
                                           burn_in = None, init_states = init_states, 
                                           return_rewards = True, seed = seed)
        Vs = [sum(r * gamma ** t for t, r in enumerate(rewards[i])) for i in range(rep)]
        threshold = 0.02
        Vs = np.clip(Vs, np.quantile(Vs, [threshold]), np.quantile(Vs, [1-threshold]))
        V_true = np.mean(Vs)
        std_true_value = np.std(Vs) / np.sqrt(len(Vs))
        if print_out:
            printR("value = {:.4f} with std {:.4f}".format(V_true, std_true_value))
        return V_true
    
    def simu_trajs_parallel(self, pi = None, N = 100, T = 1000, random_pi = True, seed = 42
                   , burn_in = None, init_states = None, return_rewards = False):
        """ parallelization to sample traj (for training and evaluation purpose)
        pi = None -> random sample
        """
        rep = N
        self.seed = seed
        ######## 
        Ss = self.reset_multiple(rep, seed = seed)
        Ss = Ss.T
        if init_states is not None:
            states = init_states.T
            Ss = init_states
        trajs = [[] for i in range(rep)]
        rewards = [[] for i in range(rep)]
        dones = [[] for i in range(rep)]
        ############
        for t in range(T):
            np.random.seed(self.seed)
            self.seed += 1
#             if t * 2 % T == 0:
#                 print("simu {}% DONE!".format(str(t / T * 100)))
            if pi is None:
                As = np.random.binomial(n = 1, p = 0.5, size = len(Ss))
            else:
                if random_pi:
                    As = pi.sample_A(Ss)
                else:
                    As = pi.get_A(Ss)
            SSs, Rs, Ds, _ = self.step_4_multiple_trajectories(As)
            for i in range(rep):
                SARS = [Ss[i].copy(), As[i], Rs[i], SSs.T[i].copy()]
                trajs[i].append(SARS)
                rewards[i].append(Rs[i])
                dones[i].append(Ds[i])
            Ss = SSs.T
        self.trajs = trajs
        ############
        if return_rewards:
            return rewards
        if burn_in is not None:
            trajs = [traj[burn_in:] for traj in trajs]
        return trajs
    
    
    def step_4_multiple_trajectories(self, actions):
        """ multiple trajectories -> vectorize!
        """
        np.random.seed(self.seed)
        self.seed += 1
        N = len(actions)
        assert self.action_space.contains(actions[0]), err_msg

        x, x_dot, theta, theta_dot = self.states # [B, 4] -> [4, N]
        force = np.array([self.force_mag if action == 1 else -self.force_mag for action in actions])
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        """ next state """
        # print(self.std, np.random.randn(4, N).T)
        errors = (np.random.randn(4, N).T * self.std).T
        
        self.old_states = self.states.copy()
        self.states = np.array([x + errors[0], x_dot + errors[1], theta + errors[2], theta_dot + errors[3]])
        
        # if done, then no change in state
        self.states[:, self.dones] = self.old_states[:, self.dones]
        
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

        
        self.dones = np.logical_or.reduce((x < -self.x_threshold, x > self.x_threshold, 
                                           theta < -self.theta_threshold_radians, 
                                           theta > self.theta_threshold_radians, self.e > self.e_max)).astype(bool)

        self.e += 1
        rewards = self.cal_reward(x, theta, with_noise = False) #1.0
        rewards[self.dones] = 0
        
        self.seed += 1
        reward_nose = self.gen_noise(u = 0, noise_std = self.reward_noise_std, size = rewards.shape, seed = self.seed)
        rewards = rewards + reward_nose
        
        return np.array(self.states), rewards, self.dones, {}

############################################################################################################################################################################################################################################################################################

    def get_init_S_from_trajs(self, trajs, n_init = 1000):
        np.random.seed(42)
        states = np.array([item[0] for traj in trajs for item in traj])
        return states[np.random.choice(len(states), n_init)]    
    
    def simu_init_S(self, seed = 42, N = None):
        if N is None:
            N = self.N
        trajs = self.simu_trajs_parallel(seed = seed, N = N, T = 2)
        

        return arr([trajs[i][0][0] for i in range(N)])

##############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    
#     def simu_one_seed(self, seed = 42, N = None, T = None, gamma = 0.95
#                      , evaluate_policy = None
#                      , evaluate_metric = None):
#         """ 
#         Simulate N trajectories with length T
#         """
#         if evaluate_metric is None: # randomly generate trajectories
#             """ 
#             """
#             trajs = self.simu_trajs_parallel(pi = evaluate_policy, rep = N, T = T)
#             return trajs
#         else:
#             """ sample traj & analyze
#             """
#             return self.eval_policy(pi = evaluate_policy, gamma = gamma, init_states = None, rep = N, T = T)


# Returns:
#     trajs = [traj], where each traj is of length T as [[S, A, R, SS]]

# actions = np.squeeze(evaluate_policy.sample_A(state.T))


# pi1 = DQN.DQN_gym(num_states = 4, num_actions = 2
#                   , hidden_units = [256, 256], gamma = gamma
#                   , gpu_number = 0)
# pi1.model.load_weights(tp_path)
# #########################################################
# pi_behav = my_gym.softmax_policy(pi1, tau)

# gym_eval = my_gym.GymEval(random_pi = True)
# trajs = gym_eval.simu_trajs_para(pi_behav, rep = 10000, burn_in = 500)

# # Evaluation a policy
# V_true = gym_eval.eval_policy(pi = pi1, gamma = gamma, init_states = init_S_4_eval, rep = len(init_S_4_eval))

# # Generate training data
# trajs = gym_eval.simu_trajs_para(pi_behav, rep = N * rep * 2, burn_in = 500)
# trajs_train_resp = [trajs[(i * N):((i + 1) * N)] for i in range(rep)]


