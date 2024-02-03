#!/bin/bash

env_list_nsac=(
	"halfcheetah-medium-v2"
	)

seed_list=(
	1
	2
	3
	4
	5
)


for seed in ${seed_list[*]}; do
for env in ${env_list_nsac[*]}; do

# noiseless cases:
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/pytorch-soft-actor-critic-master/main_SACN.py --seed=$seed --policy='BENCH' --env-name='${env_list_nsac[$i]}' --df=1.0 --std-noise=0.0
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/pytorch-soft-actor-critic-master/main_SACN.py --seed=$seed --policy='BENCH' --env-name='${env_list_nsac[$i]}' --df=1.0 --std-noise=0.0 --aggregate='MeanMStd'

# noise cases:
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/pytorch-soft-actor-critic-master/main_SACN.py --seed=$seed --policy='BENCH' --env-name='${env_list_nsac[$i]}' --df=1.0 --std-noise=2.0
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/pytorch-soft-actor-critic-master/main_SACN.py --seed=$seed --policy='BENCH' --env-name='${env_list_nsac[$i]}' --df=1.0 --std-noise=2.0 --aggregate='MeanMStd'

done
done


