class DQN(object):
    def __init__(self, num_actions=5, init_trajs = None, 
                 hiddens=[256,256], activation='relu',
                 gamma=0.99, gpu_number = 0
                 , lr=5e-4, decay_steps=100000,
                 batch_size=64, 
                 validation_split=0.2):
        """ offline??? why!!
        
        """
        ### === network ===
        self.num_A = num_actions
        self.hiddens = hiddens
        self.activation = activation
        #self.mirrored_strategy = tf.distribute.MirroredStrategy() # devices=["/gpu:0"]
        self.replay_buffer = SimpleReplayBuffer(init_trajs)
        ### === optimization ===
        self.batch_size = batch_size
        self.validation_split = validation_split
        # discount factor
        self.gamma = gamma
        
        self.target_diffs = []
        self.values = []
        self.gpu_number = gpu_number
        with tf.device('/gpu:' + str(self.gpu_number)):
            self.model = MLPNetwork(self.num_A, hiddens, activation 
                                    #, mirrored_strategy = self.mirrored_strategy
                                   )
            self.optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                                                        lr, decay_steps=decay_steps, decay_rate=1))
            self.model.compile(loss= "mse" #'huber_loss'
                               , optimizer=self.optimizer, metrics=['mse'])
        self.callbacks = []
        
        self.validation_freq = 1
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            old_targets = rewards / (1 - self.gamma)
            self.model.fit(states, old_targets, 
                           batch_size=self.batch_size, 
                           epochs= 1, 
                           verbose= 2,
                           validation_split=self.validation_split,
                           validation_freq = self.validation_freq, 
                           callbacks=self.callbacks)
    
    def fit_one_step(self, print_freq = 5, verbose = 0, nn_verbose = 0):
        with tf.device('/gpu:' + str(self.gpu_number)):
            states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
            q_next_states = self.model.predict(next_states)
            targets = rewards + self.gamma * np.max(q_next_states, 1)
            _targets = self.model.predict(states)
            _targets[range(len(actions)), actions.astype(int)] = targets
            with tf.GradientTape() as tape:
                pred_targets = self.model(states)
                loss = tf.keras.losses.MSE(_targets, pred_targets)

            dw = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(dw, self.model.trainable_variables))
            
            if verbose >= 1 and iteration % print_freq == 0:
                print('----- FQI (training) iteration: {}, target_diff = {:.3f}, values = {:.3f}'.format(iteration, target_diff, targets.mean())
                    , '-----')
    ######################
    def Q_func(self, states, actions = None):
        with tf.device('/gpu:' + str(self.gpu_number)):
            if len(states.shape) == 1:
                states = np.expand_dims(states, 0)
    #         states = (states - self.mean_S) / self.std_S
            if actions is not None:
                return np.squeeze(select_each_row(self.model(states), actions.astype(int)))
            else:
                return self.model(states)

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
            pre_dims[-1] = self.num_A
            states = states.reshape(-1, self.S_dims)
        optimal_actions = self.get_A(states)
        
        probs = np.zeros((len(states), self.num_A))
        probs[range(len(states)), optimal_actions] = 1

        if actions is None:
            if multi_dim and len(states) > 2:
                return probs.reshape(pre_dims)
            else:
                return probs
        else:
            return probs[range(len(actions)), actions]
    def sample_A(self, states):
        return self.get_A(states)
