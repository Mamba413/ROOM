env_list=(
    "halfcheetah-expert-v2"
	"walker2d-expert-v2"
	"hopper-expert-v2" 
	"halfcheetah-medium-v2"
	"walker2d-medium-v2"
	"hopper-medium-v2"
	"halfcheetah-medium-replay-v2"
	"walker2d-medium-replay-v2" 
	"hopper-medium-replay-v2"
	"halfcheetah-full-replay-v2"
	"walker2d-full-replay-v2" 
	"hopper-full-replay-v2"
	"kitchen-complete-v0"
	"kitchen-partial-v0"
	"kitchen-mixed-v0"
	)

seed_list=(
	1
	2
	3
	4
	5
)

for seed in ${seed_list[*]}; do
for env in ${env_list[*]}; do

/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/main_IQL.py --seed=$seed --policy='BENCH' --env-name='$env' --n-steps=1000000 --df=1.0 --std-noise=100.0 --ivr-alpha=10.0
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/main_IQL.py --seed=$seed --policy='MM' --aggregate='MeanMStd' --scale=0.0 --env-name='$env' --n-steps=200000 
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/main_IQL.py --seed=$seed --policy='MM' --env-name='$env' --n-steps=200000 --policy-n-steps=1000000 --df=1.0 --std-noise=100.0 --ivr-alpha=10.0
/public/home/zhujin/miniconda3/envs/mmrl/bin/python /public/home/zhujin/mmrl/MuJoCo/main_IQL.py --seed=$seed --policy='MM' --env-name='$env' --n-steps=200000 --policy-n-steps=1000000 --df=1.0 --std-noise=100.0 --ivr-alpha=10.0 --quantile=0.0

done
done

