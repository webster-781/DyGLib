# datasets=("ia-movielens-user2tags-10m" "ia-escorts-dynamic")
datasets=("ia-escorts-dynamic" "ia-digg-reply" "ia-reality-call" "ia-retweet-pol" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m")
# datasets=("ia-escorts-dynamic" )
gpu_choices=(1 1 3 2 3 0)
len=${#gpu_choices[@]}
gpu_i=0
gpu=${gpu_choices[gpu_i%len]}
init_methods=("time-exp")
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in ${init_methods[@]}; do
      nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $2 --num_runs 5 --gpu ${gpu} --optimizer AdamW --patience 20 --num_epochs 100 --load_best_configs --use_wandb $1 --negative_sample_strategy historical --position_feat_dim 64 & 
    gpu_i=$((gpu_i+1))
    gpu=${gpu_choices[gpu_i%len]}
  done
done