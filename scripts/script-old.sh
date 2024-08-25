# datasets=("ia-movielens-user2tags-10m" "ia-escorts-dynamic")
# datasets=("ia-escorts-dynamic" "ia-digg-reply" "ia-reality-call" "ia-retweet-pol" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m")
datasets=('ia-stackexch-user-marks-post-und' 'SMS-A' 'soc-sign-bitcoinalpha' 'soc-sign-bitcoinotc' 'tech-as-topology')
# datasets=("ia-escorts-dynamic" )
gpu_choices=(3 0 2 2 1)
len=${#gpu_choices[@]}
gpu_i=0
gpu=${gpu_choices[gpu_i%len]}
init_methods=("time-exp")
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in ${init_methods[@]}; do
      nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $2 --num_runs 3 --gpu ${gpu} --optimizer AdamW --patience 20 --num_epochs 50 --test_interval_epochs 5 --load_best_configs --use_wandb $1 --negative_sample_strategy historical --position_feat_dim 64 & 
    gpu_i=$((gpu_i+1))
    gpu=${gpu_choices[gpu_i%len]}
  done
done