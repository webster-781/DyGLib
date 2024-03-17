# datasets=("ia-retweet-pol" "ia-reality-call" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m" "ia-escorts-dynamic" "ia-digg-reply")
datasets=("ia-movielens-user2tags-10m" "ia-escorts-dynamic" "ia-digg-reply")
gpu=2
# init_methods=("time-exp")

# datasets=("ia-retweet-pol" "ia-reality-call")
init_methods=("time-fourier" "time-mlp2" "time-exp" "time-mlp" "time-linear")
cd /home/ayush/DyGLib/
for i in {0..2}; do
  for init_method in ${init_methods[@]}; do
    nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name TGN --num_runs 1 --gpu ${gpu} --optimizer AdamW --patience 150 --num_epochs 100 --load_best_configs --use_init_method --init_weights ${init_method} --use_wandb $1-${init_method}  & > /home/ayush/DyGLib/scripts/logs/${datasets[i]}_${init_method}
    gpu=$((gpu+1))
    gpu=$((gpu%3))
  done
done