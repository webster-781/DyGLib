# datasets=("ia-retweet-pol" "ia-reality-call" "ia-movielens-user2tags-10m" "ia-slashdot-reply-dir" "ia-escorts-dynamic" "ia-digg-reply")
# datasets=("ia-reality-call" "ia-retweet-pol" "ia-movielens-user2tags-10m")
datasets=("tech-as-topology")
# datasets=("ia-digg-reply" "ia-retweet-pol" "ia-reality-call" "ia-escorts-dynamic")
# datasets=("wikipedia" "reddit" "lastfm")
gpu_choices=(1)
len=${#gpu_choices[@]}
gpu_i=0
gpu=${gpu_choices[gpu_i%len]}
init_methods=("time-exp")
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in ${init_methods[@]}; do
    nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $1 --num_runs 3 --gpu ${gpu} --optimizer AdamW --patience 150 --num_epochs 100 --load_best_configs --use_wandb old  & > /home/ayush/DyGLib/scripts/logs/${datasets[i]}_${init_method}
    gpu_i=$((gpu_i+1))
    gpu=${gpu_choices[gpu_i%len]}
  done
done