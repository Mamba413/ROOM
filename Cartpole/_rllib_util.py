from _util import *

def get_agent_config(env_config, lr = 0.001, gamma = 0.99, n_hidden = 32):

    nn_model = dict(
            vf_share_layers= False, #True -> slow [tune vf_loss_coeff]
            fcnet_activation= 'tanh', #'tanh', #'elu', relu
            fcnet_hiddens = [n_hidden, n_hidden] #[1024, 1024] #[256, 256]
        )
    agent_config = dict(
                env_config = env_config, model = nn_model
                # uncomment the following line because it does not work when cpu and gpu is limited
                # , num_workers = 16
                , num_workers = 1
                , framework = 'tf' #'tf2'# tf2 -> slow but better performance
                , lr = lr
    #             , horizon = 43 # without this and with only self.T seems not valid, though the printed 't' and 'done' are valid?
                # The GAE (lambda) parameter.
                #, lambda = 1.0 
                # Initial coefficient for KL divergence.
                , kl_coeff = 0.2
                # Target value for KL divergence.
                , kl_target = 0.01
                , gamma = gamma
                # Coefficient of the value function loss. IMPORTANT: you must tune this if
                # you set vf_share_layers=True inside your model's config.
                , vf_loss_coeff =  0.001 #1.0 -> too large 
                , entropy_coeff = 0 # 
                 # kl_coeff: 1.0
                # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
                , num_sgd_iter = 10
                # Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
                , sgd_minibatch_size = 512 #8192
                # Number of timesteps collected for each SGD round. This defines the size of each SGD epoch.
                , train_batch_size = 4096 #32768 320000
                # PPO clip parameter.
                , clip_param = 0.3 
                # Clip param for the value function. Note that this is sensitive to the
                # scale of the rewards. If your expected V is large, increase this.
                , vf_clip_param = 200 #10.0 # these variables need to be adaptive as well. WIll change quickly
                , evaluation_interval = 1000
                , evaluation_num_episodes = int(10) # evaluation_duration
                , evaluation_duration=100
    #             , evaluation_duration = int(320 * 5)
    #             , evaluation_duration_unit =  "episodes"
                , disable_env_checking = True
                , evaluation_parallel_to_training = True
                , evaluation_config = {
                    # Example: overriding env_config, exploration, etc:
                    # "env_config": {...},
                    "explore": False}
                , 
                # uncomment the following line because it does not work when cpu and gpu is limited
                # evaluation_num_workers = 16
            )

    return agent_config