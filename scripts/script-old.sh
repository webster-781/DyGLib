datasets=("ia-movielens-user2tags-10m")
# datasets=("ia-movielens-user2tags-10m" "ia-escorts-dynamic" "ia-digg-reply")
gpu=2
init_methods=("time-exp")
cd /home/ayush/DyGLib/
for i in {0..5}; do
  for init_method in ${init_methods[@]}; do
    nohup python train_link_prediction.py --dataset_name ${datasets[i]} --model_name $1 --num_runs 1 --gpu ${gpu} --optimizer AdamW --patience 150 --num_epochs 100 --load_best_configs --use_wandb old  & > /home/ayush/DyGLib/scripts/logs/${datasets[i]}_${init_method}
    gpu=$((gpu+1))
    gpu=$((gpu%3))
  done
done