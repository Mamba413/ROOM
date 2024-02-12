Robust Offline Reinforcement Learning with Heavy-Tailed Rewards
---------------

Reproducible code for the paper: [Robust Offline Reinforcement Learning with Heavy-Tailed Rewards](https://arxiv.org/abs/2310.18715)

### Summary of the paper

This paper endeavors to augment the robustness of offline reinforcement learning (RL) in scenarios laden with heavy-tailed rewards, a prevalent circumstance in real-world applications. We propose two algorithmic frameworks, ROAM and ROOM, for robust off-policy evaluation (OPE) and offline policy optimization (OPO), respectively. Central to our frameworks is the median-of-means (MM) method. Our key insight is that employing MoM to offline RL does more than just tackle heavy-tailed rewards—it offers valid uncertainty quantification to address insufficient coverage issue in offline RL as well.

Below it is the numerical performance of our proposal (ROOM-VM & P-ROOM-VM) on the d4rl benchmarked dataset:

![](./figure/sql_bar.png)

### File structure

1. `requirement.txt`: prerequisite python libraries

2. `Cartpole` directory: code for reproducing results in Figures 3, 4, 6
   - `_density` directory: functions for estimating the density ratio in marginalize importance sampling based methods
   - `_RL` directory: employ MM in the TD update in fitted Q-iteration/evaluation based algorithms (Algorithms 4-5)
   - `_MM_OPE.py`: Algorithm 1 and its variant (ROAM-variant)
   - `_MM_OPE.py`: Algorithm 2 and its pessimistic variant (P-ROOM)
   - `_PB_OPO.py`: Bootstrap based variant for OPE.
   - `eval_cartpole.py`: reproduce Figures 3(a), 4, 6
   - `optimize_cartpole.py`: reproduce Figures 3(b)

3. `SQL`:
   - `src` directory: implement the sparse Q-learning (SQL) for 
   - `main_SQL.py`: the main file for conducting numerical studies for SQL. (reproduce Figure 5)

4. `SAC-N`:
   - `SACN.py` directory: implement the soft-actor critic (SAC) of $N$ ensemble.
   - `main_SACN.py`: the main file for conducting numerical studies for SACN. (reproduce Figure A3)
  
### Citation

```tex
@article{zhu2023robust,
  title={Robust Offline Policy Evaluation and Optimization with Heavy-Tailed Rewards},
  author={Zhu, Jin and Wan, Runzhe and Qi, Zhengling and Luo, Shikai and Shi, Chengchun},
  journal={arXiv preprint arXiv:2310.18715},
  year={2023}
}
```

### Reference

- Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization, ICLR (2023)

- Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble, NeurIPS (2021)
