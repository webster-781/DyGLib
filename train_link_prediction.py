import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import wandb
from unittest import mock
from collections import defaultdict

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.DecoLP import DecoLP
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
    set_wandb_metrics,
    find_partition_node_degrees_for_new_node_init,
)
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":
    print(torch.cuda.is_available())
    warnings.filterwarnings("ignore")
    global_log_arr = []
    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    
    total_time, time_partitioned_node_degrees = find_partition_node_degrees_for_new_node_init(dataset_name=args.dataset_name, t1_factor_of_t2=args.t1_factor_of_t2, t2_factor=0.04)
    if not args.use_init_method:
        time_partitioned_node_degrees = None
    
    # get data for training, validation and testing
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    ) = get_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0
    )
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids,
        dst_node_ids=new_node_val_data.dst_node_ids,
        seed=1,
    )
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2
    )
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        seed=3,
    )

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    (
        val_metric_all_runs,
        new_node_val_metric_all_runs,
        test_metric_all_runs,
        new_node_test_metric_all_runs,
    ) = ([], [], [], [])

    if args.use_wandb == "no":
        wandb_run = mock.Mock()
    else:
        wandb_run = wandb.init(
            entity="fb-graph-proj",
            project="fb-graph-proj-dyglib",
            config={
                "dataset": args.dataset_name,
                "run_name": args.use_wandb,
                "model": args.model_name,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "num_epochs": args.num_epochs,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "num_runs": args.num_runs,
                "negative_sample_strategy": args.negative_sample_strategy,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "time_gap": args.time_gap,
                "time_feat_dim": args.time_feat_dim,
                "num_neighbors": args.num_neighbors,
                "use_init_method": args.use_init_method,
                "t1_factor_of_t2": args.t1_factor_of_t2,
                "init_weights": args.init_weights,
                "clip_time_transformation": args.clip
            },
            group="DygLib",
            name = f"{args.model_name.lower()}-{args.dataset_name.lower()}-{args.use_wandb.lower()}"
        )

    set_wandb_metrics(wandb_run)
    with torch.autograd.set_detect_anomaly(True):
        for run in range(args.num_runs):
            set_random_seed(seed=run)

            args.seed = run
            args.save_model_name = f"{args.model_name}_seed{args.seed}_{wandb_run.name}"

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(
                f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
                exist_ok=True,
            )
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(
                f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
            )
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            # create formatter and add it to the handlers
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to logger
            logger.addHandler(fh)
            logger.addHandler(ch)

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f"configuration is {args}")

            # create model
            if args.model_name == "TGAT":
                dynamic_backbone = TGAT(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                (
                    src_node_mean_time_shift,
                    src_node_std_time_shift,
                    dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift,
                ) = compute_src_dst_node_time_shifts(
                    train_data.src_node_ids,
                    train_data.dst_node_ids,
                    train_data.node_interact_times,
                )
                dynamic_backbone = MemoryModel(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    model_name=args.model_name,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    src_node_mean_time_shift=src_node_mean_time_shift,
                    src_node_std_time_shift=src_node_std_time_shift,
                    dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift=dst_node_std_time_shift,
                    device=args.device,
                    time_partitioned_node_degrees = time_partitioned_node_degrees,
                    total_time = total_time,
                    init_weights = args.init_weights
                )
            elif args.model_name == "DecoLP":
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                (
                    src_node_mean_time_shift,
                    src_node_std_time_shift,
                    dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift,
                ) = compute_src_dst_node_time_shifts(
                    train_data.src_node_ids,
                    train_data.dst_node_ids,
                    train_data.node_interact_times,
                )
                dynamic_backbone = DecoLP(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    wandb_run = wandb_run,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    src_node_mean_time_shift=src_node_mean_time_shift,
                    src_node_std_time_shift=src_node_std_time_shift,
                    dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                    dst_node_std_time_shift=dst_node_std_time_shift,
                    device=args.device,
                    save_prev=args.num_neighbors,
                    use_ROPe = args.use_ROPe,
                    position_feat_dim = args.position_feat_dim
                )

            elif args.model_name == "CAWN":
                dynamic_backbone = CAWN(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    position_feat_dim=args.position_feat_dim,
                    walk_length=args.walk_length,
                    num_walk_heads=args.num_walk_heads,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "TCL":
                dynamic_backbone = TCL(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    num_depths=args.num_neighbors + 1,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "GraphMixer":
                dynamic_backbone = GraphMixer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    num_tokens=args.num_neighbors,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    device=args.device,
                )
            elif args.model_name == "DyGFormer":
                dynamic_backbone = DyGFormer(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=train_neighbor_sampler,
                    time_feat_dim=args.time_feat_dim,
                    channel_embedding_dim=args.channel_embedding_dim,
                    patch_size=args.patch_size,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    max_input_sequence_length=args.max_input_sequence_length,
                    device=args.device,
                )
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            link_predictor = MergeLayer(
                input_dim1=args.position_feat_dim,
                input_dim2=args.position_feat_dim,
                hidden_dim=args.position_feat_dim,
                output_dim=1,
            )
            model = nn.Sequential(dynamic_backbone, link_predictor)
            logger.info(f"model -> {model}")
            logger.info(
                f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
                f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
            )

            optimizer = create_optimizer(
                model=model,
                optimizer_name=args.optimizer,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            model = convert_to_gpu(model, device=args.device)

            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

            early_stopping = EarlyStopping(
                patience=args.patience,
                save_model_folder=save_model_folder,
                save_model_name=args.save_model_name,
                logger=logger,
                model_name=args.model_name,
            )

            loss_func = nn.BCELoss()
            wandb_run.watch(model, 50)
            for epoch in range(args.num_epochs):
                model.train()
                if args.model_name in [
                    "DyRep",
                    "TGAT",
                    "TGN",
                    "CAWN",
                    "TCL",
                    "GraphMixer",
                    "DyGFormer",
                ]:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()

                # store train losses and metrics
                train_losses, train_metrics = [], []
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    wandb_log_dict = {}
                    train_data_indices = train_data_indices.numpy()
                    (
                        batch_src_node_ids,
                        batch_dst_node_ids,
                        batch_node_interact_times,
                        batch_edge_ids,
                    ) = (
                        train_data.src_node_ids[train_data_indices],
                        train_data.dst_node_ids[train_data_indices],
                        train_data.node_interact_times[train_data_indices],
                        train_data.edge_ids[train_data_indices],
                    )

                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                        size=len(batch_src_node_ids)
                    )
                    batch_neg_src_node_ids = batch_src_node_ids

                    # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                    # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                    if args.model_name in ["TGAT", "CAWN", "TCL"]:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[
                            0
                        ].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                        )

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        (
                            batch_neg_src_node_embeddings,
                            batch_neg_dst_node_embeddings,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                        )
                    elif args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                        wandb_log_dict['all_emb_mean'] = torch.mean(model[0].memory_bank.node_memories)
                        wandb_log_dict['all_emb_std'] = torch.std(model[0].memory_bank.node_memories)

                        # note that negative nodes do not change the memories while the positive nodes change the memories,
                        # we need to first compute the embeddings of negative nodes for memory-based models
                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        (
                            batch_neg_src_node_embeddings,
                            batch_neg_dst_node_embeddings,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=None,
                            edges_are_positive=False,
                            num_neighbors=args.num_neighbors,
                            log_dict = wandb_log_dict
                        )

                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[
                            0
                        ].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=batch_edge_ids,
                            edges_are_positive=True,
                            num_neighbors=args.num_neighbors,
                            log_dict = wandb_log_dict
                        )
                    elif args.model_name in ["GraphMixer"]:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[
                            0
                        ].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        (
                            batch_neg_src_node_embeddings,
                            batch_neg_dst_node_embeddings,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )
                    elif args.model_name in ["DyGFormer"]:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = model[
                            0
                        ].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                        )

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        (
                            batch_neg_src_node_embeddings,
                            batch_neg_dst_node_embeddings,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                        )
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                    # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = (
                        model[1](
                            input_1=batch_src_node_embeddings,
                            input_2=batch_dst_node_embeddings,
                        )
                        .squeeze(dim=-1)
                        .sigmoid()
                    )
                    negative_probabilities = (
                        model[1](
                            input_1=batch_neg_src_node_embeddings,
                            input_2=batch_neg_dst_node_embeddings,
                        )
                        .squeeze(dim=-1)
                        .sigmoid()
                    )

                    predicts = torch.cat(
                        [positive_probabilities, negative_probabilities], dim=0
                    )
                    labels = torch.cat(
                        [
                            torch.ones_like(positive_probabilities),
                            torch.zeros_like(negative_probabilities),
                        ],
                        dim=0,
                    )
                    
                    loss = loss_func(input=predicts, target=labels)

                    train_losses.append(loss.item())

                    train_metrics.append(
                        get_link_prediction_metrics(predicts=predicts, labels=labels)
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
                    optimizer.step()
                    for name, param in model.named_parameters():
                        if param.data.isnan().any() or (param.requires_grad and param.grad is not None and param.grad.isnan().any()):
                            breakpoint()

                    train_idx_data_loader_tqdm.set_description(
                        f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
                    )

                    if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                        # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                        model[0].memory_bank.detach_memory_bank()

                if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                    # backup memory bank after training so it can be used for new validation nodes
                    train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                val_losses, val_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

                if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                    # reload training memory bank for new validation nodes
                    model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

                new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                    evaluate_data=new_node_val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

                if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(
                    f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.nanmean(train_losses):.4f}'
                )
                wandb_log_dict["train_loss"] = np.nanmean(train_losses)
                for metric_name in train_metrics[0].keys():
                    logger.info(
                        f"train {metric_name}, {np.nanmean([train_metric[metric_name] for train_metric in train_metrics]):.4f}"
                    )
                    wandb_log_dict[f"train {metric_name}"] = np.nanmean(
                        [train_metric[metric_name] for train_metric in train_metrics]
                    )
                logger.info(f"validate loss: {np.nanmean(val_losses):.4f}")
                wandb_log_dict["val_loss"] = np.nanmean(val_losses)
                for metric_name in val_metrics[0].keys():
                    logger.info(
                        f"validate {metric_name}, {np.nanmean([val_metric[metric_name] for val_metric in val_metrics]):.4f}"
                    )
                    wandb_log_dict[f"val {metric_name}"] = np.nanmean(
                        [train_metric[metric_name] for train_metric in train_metrics]
                    )
                logger.info(f"new node validate loss: {np.nanmean(new_node_val_losses):.4f}")
                wandb_log_dict["new node val_loss"] = np.nanmean(new_node_val_losses)
                for metric_name in new_node_val_metrics[0].keys():
                    logger.info(
                        f"new node validate {metric_name}, {np.nanmean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}"
                    )
                    wandb_log_dict[f"new node val {metric_name}"] = np.nanmean(
                        [
                            new_node_val_metric[metric_name]
                            for new_node_val_metric in new_node_val_metrics
                        ]
                    )

                # perform testing once after test_interval_epochs
                if (epoch + 1) % args.test_interval_epochs == 0:
                    test_losses, test_metrics = evaluate_model_link_prediction(
                        model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=test_idx_data_loader,
                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
                        evaluate_data=test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                        # reload validation memory bank for new testing nodes
                        model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                    (
                        new_node_test_losses,
                        new_node_test_metrics,
                    ) = evaluate_model_link_prediction(
                        model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=new_node_test_idx_data_loader,
                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                        evaluate_data=new_node_test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                        # reload validation memory bank for testing nodes or saving models
                        # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                        model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                    logger.info(f"test loss: {np.nanmean(test_losses):.4f}")
                    wandb_log_dict["test_loss"] = np.nanmean(test_losses)
                    for metric_name in test_metrics[0].keys():
                        logger.info(
                            f"test {metric_name}, {np.nanmean([test_metric[metric_name] for test_metric in test_metrics]):.4f}"
                        )
                        wandb_log_dict[f"test {metric_name}"] = np.nanmean(
                            [test_metric[metric_name] for test_metric in test_metrics]
                        )
                    logger.info(f"new node test loss: {np.nanmean(new_node_test_losses):.4f}")
                    wandb_log_dict["new node test loss"] = np.nanmean(new_node_test_losses)
                    for metric_name in new_node_test_metrics[0].keys():
                        logger.info(
                            f"new node test {metric_name}, {np.nanmean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}"
                        )
                        wandb_log_dict[f"new node test {metric_name}"] = np.nanmean(
                            [
                                new_node_test_metric[metric_name]
                                for new_node_test_metric in new_node_test_metrics
                            ]
                        )
                if args.model_name == 'DecoLP':
                    wandb_log_dict['avg_ff_weight_norm'] = torch.sum(torch.tensor([torch.norm(dynamic_backbone.memory_updater.memory_updater.encoder.layers[i].linear1.weight) + torch.norm(dynamic_backbone.memory_updater.memory_updater.encoder.layers[i].linear2.weight) for i in range(dynamic_backbone.memory_updater.memory_updater.encoder.num_layers)]))
                wandb_run.log(wandb_log_dict, commit = True)
                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append(
                        (
                            metric_name,
                            np.nanmean(
                                [val_metric[metric_name] for val_metric in val_metrics]
                            ),
                            True,
                        )
                    )
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break

            # load the best model
            early_stopping.load_checkpoint(model)

            # evaluate the best model
            logger.info(f"get final performance on dataset {args.dataset_name}...")

            # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
            if args.model_name not in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                val_losses, val_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_neg_edge_sampler=val_neg_edge_sampler,
                    evaluate_data=val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

                new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                    evaluate_data=new_node_val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

            if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            test_losses, test_metrics = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader,
                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                evaluate_data=test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

            if args.model_name in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                # reload validation memory bank for new testing nodes
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_test_idx_data_loader,
                evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                evaluate_data=new_node_test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )
            # store the evaluation metrics at the current run
            (
                val_metric_dict,
                new_node_val_metric_dict,
                test_metric_dict,
                new_node_test_metric_dict,
            ) = ({}, {}, {}, {})

            if args.model_name not in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                logger.info(f"validate loss: {np.nanmean(val_losses):.4f}")
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.nanmean(
                        [val_metric[metric_name] for val_metric in val_metrics]
                    )
                    logger.info(f"validate {metric_name}, {average_val_metric:.4f}")
                    val_metric_dict[metric_name] = average_val_metric

                logger.info(f"new node validate loss: {np.nanmean(new_node_val_losses):.4f}")
                for metric_name in new_node_val_metrics[0].keys():
                    average_new_node_val_metric = np.nanmean(
                        [
                            new_node_val_metric[metric_name]
                            for new_node_val_metric in new_node_val_metrics
                        ]
                    )
                    logger.info(
                        f"new node validate {metric_name}, {average_new_node_val_metric:.4f}"
                    )
                    new_node_val_metric_dict[metric_name] = average_new_node_val_metric

            logger.info(f"test loss: {np.nanmean(test_losses):.4f}")
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.nanmean(
                    [test_metric[metric_name] for test_metric in test_metrics]
                )
                logger.info(f"test {metric_name}, {average_test_metric:.4f}")
                test_metric_dict[metric_name] = average_test_metric

            logger.info(f"new node test loss: {np.nanmean(new_node_test_losses):.4f}")
            for metric_name in new_node_test_metrics[0].keys():
                average_new_node_test_metric = np.nanmean(
                    [
                        new_node_test_metric[metric_name]
                        for new_node_test_metric in new_node_test_metrics
                    ]
                )
                logger.info(
                    f"new node test {metric_name}, {average_new_node_test_metric:.4f}"
                )
                new_node_test_metric_dict[metric_name] = average_new_node_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

            if args.model_name not in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                val_metric_all_runs.append(val_metric_dict)
                new_node_val_metric_all_runs.append(new_node_val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)
            new_node_test_metric_all_runs.append(new_node_test_metric_dict)

            # avoid the overlap of logs
            if run < args.num_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            if args.model_name not in ["JODIE", "DyRep", "TGN", "DecoLP"]:
                result_json = {
                    "validate metrics": {
                        metric_name: f"{val_metric_dict[metric_name]:.4f}"
                        for metric_name in val_metric_dict
                    },
                    "new node validate metrics": {
                        metric_name: f"{new_node_val_metric_dict[metric_name]:.4f}"
                        for metric_name in new_node_val_metric_dict
                    },
                    "test metrics": {
                        metric_name: f"{test_metric_dict[metric_name]:.4f}"
                        for metric_name in test_metric_dict
                    },
                    "new node test metrics": {
                        metric_name: f"{new_node_test_metric_dict[metric_name]:.4f}"
                        for metric_name in new_node_test_metric_dict
                    },
                }
            else:
                result_json = {
                    "test metrics": {
                        metric_name: f"{test_metric_dict[metric_name]:.4f}"
                        for metric_name in test_metric_dict
                    },
                    "new node test metrics": {
                        metric_name: f"{new_node_test_metric_dict[metric_name]:.4f}"
                        for metric_name in new_node_test_metric_dict
                    },
                }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(
                save_result_folder, f"{args.save_model_name}.json"
            )
            with open(save_result_path, "w") as file:
                file.write(result_json)
            wandb_run.save(save_result_path)
            wandb_run.finish()

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {args.num_runs} runs:")

    if args.model_name not in ["JODIE", "DyRep", "TGN", "DecoLP"]:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f"validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}"
            )
            logger.info(
                f"average validate {metric_name}, {np.nanmean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} "
                f"± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}"
            )

        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(
                f"new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}"
            )
            logger.info(
                f"average new node validate {metric_name}, {np.nanmean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} "
                f"± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}"
            )

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f"test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}"
        )
        logger.info(
            f"average test {metric_name}, {np.nanmean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} "
            f"± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}"
        )

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(
            f"new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}"
        )
        logger.info(
            f"average new node test {metric_name}, {np.nanmean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} "
            f"± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}"
        )

    sys.exit()
