# datasets=("ia-escorts-dynamic" "ia-digg-reply" "ia-reality-call" "ia-retweet-pol" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m")
datasets=('ia-stackexch-user-marks-post-und' 'SMS-A' 'soc-sign-bitcoinalpha' 'soc-sign-bitcoinotc' 'tech-as-topology')
# datasets=("ia-escorts-dynamic" "ia-movielens-user2tags-10m")
gpu_choices=(0 1 3 3 2)
len=${#gpu_choices[@]}
declare -a init_methods=(
                 "time-linear time-exp"
                )

# init_methods=("time-fourier" "time-mlp2" "time-exp" "time-mlp" "time-linear")
gpu_i=0
gpu=${gpu_choices[gpu_i%len]}
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in "${init_methods[@]}"
    do
    nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $2 --num_runs 3 --gpu ${gpu} --optimizer AdamW --patience 20 --num_epochs 50 --test_interval_epochs 5 --load_best_configs --use_init_method --t1_factor_of_t2 3 --init_weights ${init_method} --use_wandb $1 --attfus --num_combinations 32 --num_samples_per_combination 200 --negative_sample_strategy historical --position_feat_dim 64 --predictor mlp --last_k 20 & > /home/ayush/DyGLib/scripts/logs/${datasets[i]}_$1 
    gpu_i=$((gpu_i+1))
    gpu=${gpu_choices[gpu_i%len]}
  done
done