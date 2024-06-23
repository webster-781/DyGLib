datasets=("ia-escorts-dynamic" "ia-digg-reply" "ia-reality-call" "ia-retweet-pol" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m")
# datasets=("ia-retweet-pol" "ia-escorts-dynamic" "ia-movielens-user2tags-10m")
# datasets=("ia-escorts-dynamic" "ia-digg-reply")
gpu_choices=(0 2 1 2 1 0)
len=${#gpu_choices[@]}
declare -a init_methods=(
                 "time-exp time-linear"
                )

# init_methods=("time-fourier" "time-mlp2" "time-exp" "time-mlp" "time-linear")
gpu_i=0
gpu=${gpu_choices[gpu_i%len]}
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in "${init_methods[@]}"
    do
    nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $2 --num_runs 5 --gpu ${gpu} --optimizer AdamW --patience 150 --num_epochs 100 --load_best_configs --use_init_method --t1_factor_of_t2 3 --init_weights ${init_method} --use_wandb $1 --attfus --num_combinations 64 --num_samples_per_combination 400  & > /home/ayush/DyGLib/scripts/logs/${datasets[i]}_$1
    gpu_i=$((gpu_i+1))
    gpu=${gpu_choices[gpu_i%len]}
  done
done