# datasets=("ia-retweet-pol" "ia-reality-call" "ia-slashdot-reply-dir" "ia-movielens-user2tags-10m" "ia-escorts-dynamic" "ia-digg-reply")
datasets=("ia-retweet-pol")

# init_methods=("time-exp" "time-linear" "time-fourier" "time-mlp")
init_methods=("time-exp" "time-linear")

for dataset_name in ${datasets[@]}; do
  for init_method in ${init_methods[@]}; do
    nohup python /home/ayush/DyGLib/train_link_prediction.py --dataset_name ia-slashdot-reply-dir --model_name TGN --num_runs 1 --gpu 1 --optimizer AdamW --patience 150 --num_epochs 100 --load_best_configs --use_init_method --init_weights ${init_method} --use_wandb reparam-${init_method}
 > ${dataset_name}_${init_method}
  done
done