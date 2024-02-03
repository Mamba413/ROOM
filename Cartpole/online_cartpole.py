# %%
import ray
from ray.rllib.agents import ppo
import _rllib_util
import gym

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# %%
gamma = 0.99
lr = 0.001
n_hidden_nodes=32

# %%
ray.init(num_gpus=1, num_cpus=2)  
agent_config = _rllib_util.get_agent_config(env_config=dict(), lr=lr, gamma=gamma, n_hidden=n_hidden_nodes,)
trainer = ppo.PPOTrainer(env="CartPole-v0", config=agent_config)
mean_training_rewards = []
for i in range(20):
    res = trainer.train()
    print(res['episode_reward_mean'])

# %%
checkpoint = trainer.save('cp/cartpole_gamma_{}'.format(gamma))
print("checkpoint saved at", checkpoint)

