datasets=("myket"  "mooc" "wikipedia" "reddit" "enron" "myket")

for dataset_name in ${datasets[@]}; do
  python train_link_prediction.py --dataset_name $dataset_name --model_name DecoLP --num_runs 1 --gpu 0 --num_layers 2 --num_heads 2 --dropout 0.1 --time_feat_dim 4 --use_wandb new_iterations --optimizer AdamW --learning_rate 0.0001 --weight_decay 0.05 --patience 50 --num_epochs 200 --use_wandb test_all --num_neighbors 150 --use_ROPe --position_feat_dim 96
done