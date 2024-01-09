import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
import math

from utils.utils import NeighborSampler, vectorized_update_mem_3d
from models.modules import TimeEncoder
from models.MemoryModel import MessageAggregator


class DecoLP(torch.nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_feat_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        src_node_mean_time_shift: float = 0.0,
        src_node_std_time_shift: float = 1.0,
        dst_node_mean_time_shift_dst: float = 0.0,
        dst_node_std_time_shift: float = 1.0,
        save_prev: int = 50,
        device: str = "cpu",
    ):
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(DecoLP, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift
        self.memory_dim = self.node_feat_dim
        self.save_prev = save_prev
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = (
            self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim
        )

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregatorDecoLP()

        # memory modules
        self.memory_bank = MemoryBankDecoLP(
            num_nodes=self.num_nodes,
            memory_dim=self.memory_dim,
            transformer_dim=self.message_dim,
            save_prev=self.save_prev,
        )
        self.memory_updater = MemoryUpdaterDecoLP(
            self.memory_bank,
            num_heads=self.num_heads,
            memory_dim=self.memory_dim,
            num_layers=self.num_layers,
            transformer_dim=self.message_dim,
            dropout=self.dropout,
            save_prev=self.save_prev,
        )

        # embedding module
        self.embedding_module = TimeProjectionEmbeddingDecoLP(
            memory_dim=self.memory_dim, dropout=self.dropout
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        edges_are_positive: bool = True,
        num_neighbors: int = 20,
    ):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # so we return all nodes' memory with shape (num_nodes, ) by using self.get_updated_memory
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        (
            updated_node_memories,
            updated_node_last_updated_times,
        ) = self.get_updated_memories(
            node_ids=node_ids,
            node_raw_messages=self.memory_bank.node_raw_messages,
        )
        
        # compute the node temporal embeddings using the embedding module
        # compute differences between the time the memory of a node was last updated, and the time for which we want to compute the embedding of a node
        # Tensor, shape (batch_size, )
        src_node_time_intervals = (
            torch.from_numpy(node_interact_times).float().to(self.device)
            - updated_node_last_updated_times[torch.from_numpy(src_node_ids)]
        )
        src_node_time_intervals = (
            src_node_time_intervals - self.src_node_mean_time_shift
        ) / self.src_node_std_time_shift
        # Tensor, shape (batch_size, )
        dst_node_time_intervals = (
            torch.from_numpy(node_interact_times).float().to(self.device)
            - updated_node_last_updated_times[torch.from_numpy(dst_node_ids)]
        )
        dst_node_time_intervals = (
            dst_node_time_intervals - self.dst_node_mean_time_shift_dst
        ) / self.dst_node_std_time_shift
        # Tensor, shape (2 * batch_size, )
        node_time_intervals = torch.cat(
            [src_node_time_intervals, dst_node_time_intervals], dim=0
        )
        # Tensor, shape (2 * batch_size, memory_dim), which is equal to (2 * batch_size, node_feat_dim)
        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
            node_memories=updated_node_memories,
            node_ids=node_ids,
            node_time_intervals=node_time_intervals,
        )

        # two Tensors, with shape (batch_size, node_feat_dim)
        src_node_embeddings, dst_node_embeddings = (
            node_embeddings[: len(src_node_ids)],
            node_embeddings[len(src_node_ids) : len(src_node_ids) + len(dst_node_ids)],
        )

        if edges_are_positive:
            assert edge_ids is not None
            # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
            self.update_memories(
                node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages
            )

            # clear raw messages for source and destination nodes since we have already updated the memory using them
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            # compute new raw messages for source and destination nodes
            (
                unique_src_node_ids,
                new_src_node_raw_messages,
            ) = self.compute_new_node_raw_messages(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                dst_node_embeddings=dst_node_embeddings,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids,
            )
            (
                unique_dst_node_ids,
                new_dst_node_raw_messages,
            ) = self.compute_new_node_raw_messages(
                src_node_ids=dst_node_ids,
                dst_node_ids=src_node_ids,
                dst_node_embeddings=src_node_embeddings,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids,
            )

            # store new raw messages for source and destination nodes
            self.memory_bank.store_node_raw_messages(
                node_ids=unique_src_node_ids,
                new_node_raw_messages=new_src_node_raw_messages,
            )
            self.memory_bank.store_node_raw_messages(
                node_ids=unique_dst_node_ids,
                new_node_raw_messages=new_dst_node_raw_messages,
            )

        return src_node_embeddings, dst_node_embeddings

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        get the updated memories based on node_ids and node_raw_messages (just for computation), but not update the memories
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        (
            unique_node_ids,
            to_update_node_ids,
            unique_node_messages,
            unique_node_timestamps,
            to_not_update_node_ids,
            unique_inverse_indices
        ) = self.message_aggregator.aggregate_messages(
            node_ids=node_ids, node_raw_messages=node_raw_messages
        )
        # get updated memory for all nodes with messages stored in previous batches (just for computation)
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        (
            updated_node_memories,
            updated_node_last_updated_times,
        ) = self.memory_updater.get_updated_memories(
            all_node_ids = node_ids,
            unique_node_ids=unique_node_ids,
            to_update_node_ids = to_update_node_ids,
            to_not_update_node_ids = to_not_update_node_ids, 
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps,
            unique_inverse_indices = unique_inverse_indices,
        )

        return updated_node_memories, updated_node_last_updated_times

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        (
            unique_node_ids,
            to_update_node_ids,
            unique_node_messages,
            unique_node_timestamps,
            to_not_update_node_ids,
            unique_inverse_indices
        ) = self.message_aggregator.aggregate_messages(
            node_ids=node_ids, node_raw_messages=node_raw_messages
        )

        # update the memories with the aggregated messages
        self.memory_updater.update_memories(
            unique_node_ids=unique_node_ids[to_update_node_ids],
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps,
        )

    def compute_new_node_raw_messages(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        dst_node_embeddings: torch.Tensor,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
    ):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, save_prev, memory_dim)
        src_node_embeddings = self.memory_bank.get_embeddings(node_ids=src_node_ids)
        dst_node_embeddings = self.memory_bank.get_embeddings(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = (
            torch.from_numpy(node_interact_times).float().to(self.device)
            - self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        )

        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(
            src_node_delta_times.unsqueeze(dim=1)
        ).reshape(len(src_node_ids), -1)

        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat(
            [
                src_node_embeddings,
                dst_node_embeddings,
                src_node_delta_time_features,
                edge_features,
            ],
            dim=1,
        )

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append(
                (new_src_node_raw_messages[i], node_interact_times[i])
            )

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in [
            "TGN",
            "DyRep",
        ], f"Neighbor sampler is not defined in model {self.model_name}!"
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in [
            "uniform",
            "time_interval_aware",
        ]:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()


# Memory-related Modules
class MemoryBankDecoLP(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        transformer_dim: int,
        save_prev: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBankDecoLP, self).__init__()
        self.num_nodes = num_nodes
        self.save_prev = save_prev
        self.memory_dim = memory_dim
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(
            torch.zeros((self.num_nodes, self.save_prev, self.transformer_dim)),
            requires_grad=False,
        )
        # Parameter, which contains the number of updates for each node until this moment
        self.node_num_updates = nn.Parameter(
            torch.zeros(self.num_nodes, dtype=int), requires_grad=False
        )
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(
            torch.zeros(self.num_nodes), requires_grad=False
        )
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_num_updates.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def get_embeddings(self, node_ids: np.ndarray):
        """
        get each nodes embeddings any some point in time
        this embedding is actually nothing but the first `dim` dimensions of the latest transformer ouptut for that layer
        """
        ids = torch.from_numpy(node_ids)
        return self.node_memories[ids, self.node_num_updates[ids], : self.memory_dim]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, transformer_dim)
        :return:
        """
        (
            self.node_memories[torch.from_numpy(node_ids)],
            self.node_num_updates,
        ) = vectorized_update_mem_3d(
            self.node_memories[torch.from_numpy(node_ids)],
            self.node_num_updates,
            updated_node_memories,
        )

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [
                (node_raw_message[0].clone(), node_raw_message[1].copy())
                for node_raw_message in node_raw_messages
            ]

        return (
            self.node_memories.data.clone(),
            self.node_last_updated_times.data.clone(),
            self.node_num_updates.clone(),
            cloned_node_raw_messages,
        )

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        (
            self.node_memories.data,
            self.node_last_updated_times.data,
            self.node_num_updates.data,
        ) = (
            backup_memory_bank[0].clone(),
            backup_memory_bank[1].clone(),
            backup_memory_bank[2].clone(),
        )

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[3].items():
            self.node_raw_messages[node_id] = [
                (node_raw_message[0].clone(), node_raw_message[1].copy())
                for node_raw_message in node_raw_messages
            ]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append(
                    (node_raw_message[0].detach(), node_raw_message[1])
                )

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(
        self, node_ids: np.ndarray, new_node_raw_messages: dict
    ):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return "num_nodes={}, memory_dim={}".format(
            self.node_memories.shape[0], self.node_memories.shape[1]
        )


class MemoryUpdaterDecoLP(nn.Module):
    def __init__(
        self,
        memory_bank: MemoryBankDecoLP,
        num_heads: int,
        memory_dim: int,
        num_layers: int,
        transformer_dim: int,
        dropout: float,
        save_prev: int,
    ):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdaterDecoLP, self).__init__()
        self.memory_bank = memory_bank
        self.num_heads = num_heads
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.save_prev = save_prev
        self.memory_dim = memory_dim
        self.memory_updater = MemoryUpdaterModule(
            transformer_dim=self.transformer_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            memory_dim=self.memory_dim,
        )

    def update_memories(
        self,
        unique_node_ids: np.ndarray,
        unique_node_messages: torch.Tensor,
        unique_node_timestamps: np.ndarray,
    ):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, transformer_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return

        assert (
            (
                self.memory_bank.get_node_last_updated_times(unique_node_ids)
                <= torch.from_numpy(unique_node_timestamps)
                .float()
                .to(unique_node_messages.device)
            )
            .all()
            .item()
        ), "Trying to update memory to time in the past!"
        num_unique_node_ids = unique_node_ids.shape[0]
        # Tensor, shape (num_unique_node_ids, save_prev, transformer_dim)
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids).clone()
        # Tensor, shape (num_unique_node_ids,)
        node_num_updates = self.memory_bank.node_num_updates[unique_node_ids].clone()
        # Tensor, shape (num_unique_node_ids, save_prev + 1, transformer_dim)
        node_memories = torch.stack(
            (node_memories, torch.zeros(num_unique_node_ids, 1, self.transformer_dim)),
            dim=1,
        )

        batch_indices = torch.arange(num_unique_node_ids)  # Tensor [0, 1, 2, ..., n-1]
        node_memories[batch_indices, node_num_updates] = unique_node_messages
        # Tensor, shape (num_unique_node_ids, save_prev+1, transformer_dim)
        output = self.memory_updater(node_memories)

        # Tensor, shape (num_unique_node_ids, memory_dim)
        updated_node_memories = output[batch_indices, node_num_updates]

        # updated_node_memories should contain [u|i|t|f]    --> u is the node1 embedding
        #                                                   --> i is the node2 embedding
        updated_node_memories = torch.stack(
            (updated_node_memories, unique_node_messages[:, self.memory_dim:]), dim=1
        )

        # update memories for nodes in unique_node_ids
        self.memory_bank.set_memories(
            node_ids=unique_node_ids, updated_node_memories=updated_node_memories
        )

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = (
            torch.from_numpy(unique_node_timestamps)
            .float()
            .to(unique_node_messages.device)
        )

    def get_updated_memories(
        self,
        all_node_ids: np.ndarray,
        unique_node_ids: np.ndarray,
        unique_node_messages: torch.Tensor,
        unique_node_timestamps: np.ndarray,
        to_update_node_ids: np.ndarray,
        to_not_update_node_ids: np.ndarray,
        unique_inverse_indices: np.ndarray
    ):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        For nodes to be updated, pass through the transformer. For nodes to not be updated, just take the embeddings.
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return (
                self.memory_bank.node_memories.data[:, -1, : self.memory_dim].clone(),
                self.memory_bank.node_last_updated_times.data.clone(),
            )

        assert (
            (
                self.memory_bank.get_node_last_updated_times(
                    unique_node_ids=unique_node_ids
                )
                <= torch.from_numpy(unique_node_timestamps)
                .float()
                .to(unique_node_messages.device)
            )
            .all()
            .item()
        ), "Trying to update memory to time in the past!"

        num_unique_node_ids = unique_node_ids.shape[0]
        num_to_update_unique_node_ids = to_update_node_ids.shape[0]
        num_to_not_update_unique_node_ids = to_update_node_ids.shape[0]
        # Tensor, shape (num_unique_node_ids, save_prev, transformer_dim)
        node_memories= self.memory_bank.get_memories(node_ids=unique_node_ids).clone()
        # Tensor, shape (num_unique_node_ids,)
        
        node_memories_to_update = node_memories[torch.from_numpy(to_update_node_ids)]
        node_num_updates = self.memory_bank.node_num_updates[all_node_ids].clone()
        node_to_update_num_updates = node_num_updates[to_update_node_ids]
        node_to_not_update_num_updates = node_num_updates[to_not_update_node_ids]
            
        # Tensor, shape (num_unique_node_ids, save_prev + 1, transformer_dim)
        node_memories_to_update = torch.cat(
            (
                node_memories_to_update,
                torch.zeros(to_update_node_ids.shape[0], 1, self.transformer_dim).to(
                    node_memories_to_update.device
                ),
            ),
            dim=1,
        )

        batch_indices_to_update = torch.arange(num_to_update_unique_node_ids)  # Tensor [0, 1, 2, ..., n-1]
        batch_indices_to_not_update = torch.arange(num_to_not_update_unique_node_ids)  # Tensor [0, 1, 2, ..., n-1]

        # For nodes to be updated, do the update
        node_memories_to_update[batch_indices_to_update, node_to_update_num_updates] = unique_node_messages
        output = self.memory_updater(node_memories_to_update)

        node_memories_to_update = output[batch_indices_to_update, node_to_update_num_updates]
        
        # For nodes to not be updated, 
        node_memories_to_not_update = node_memories[batch_indices_to_not_update, node_to_not_update_num_updates, :self.memory_dim]

        new_node_memories = torch.zeros(num_unique_node_ids, self.memory_dim)
        new_node_memories[to_update_node_ids] += node_memories_to_update
        new_node_memories[to_not_update_node_ids] += node_memories_to_not_update
        
        new_node_memories_for_all_nodes = new_node_memories[unique_inverse_indices]
        
        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = (
            self.memory_bank.node_last_updated_times.data.clone()
        )
        
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids[to_update_node_ids])] = (
            torch.from_numpy(unique_node_timestamps)
            .float()
            .to(unique_node_messages.device)
        )

        return new_node_memories_for_all_nodes, updated_node_last_updated_times


# Embedding-related Modules
class TimeProjectionEmbeddingDecoLP(nn.Module):
    def __init__(self, memory_dim: int, dropout: float):
        """
        Time projection embedding module.
        :param memory_dim: int, dimension of node memories
        :param dropout: float, dropout rate
        """
        super(TimeProjectionEmbeddingDecoLP, self).__init__()

        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)

        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(
        self,
        node_memories: torch.Tensor,
        node_ids: np.ndarray,
        node_time_intervals: torch.Tensor,
    ):
        """
        compute node temporal embeddings using the embedding projection operation in JODIE
        :param node_memories: Tensor, shape (num_nodes, transformer_dim)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_time_intervals: Tensor, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        breakpoint()
        if node_memories.shape[0] > node_ids.shape[0]:
            node_memories = node_memories[torch.from_numpy(node_ids)]
        source_embeddings = self.dropout(
            node_memories
            * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1)))
        )

        return source_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MemoryUpdaterModule(nn.Module):
    def __init__(self, transformer_dim, num_heads, num_layers, dropout, memory_dim):
        super(MemoryUpdaterModule, self).__init__()
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.memory_dim = memory_dim
        self.norm = nn.LayerNorm(self.transformer_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.transformer_dim, nhead=self.num_heads, dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=self.num_layers, norm=self.norm
        )
        self.pos_encoder = PositionalEncoding(self.transformer_dim, self.dropout)
        self.output_layer = nn.Linear(
            in_features=self.transformer_dim, out_features=self.memory_dim, bias=True
        )

    def forward(self, x: torch.tensor):
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        seq_len = x.size(0)  # Sequence length
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(
            x.device
        )

        # For simplicity, let's assume 'tgt' is initially the same as 'mem'
        # but shifted by one position (you might start with a special start token)
        tgt = torch.roll(x, shifts=-1, dims=1)

        # Apply causal mask (assuming you're doing autoregressive generation)
        x = self.decoder(tgt, x, tgt_mask=mask)
        x = x.permute(1, 0, 2)
        x = self.output_layer(x)
        return x



# Message-related Modules
class MessageAggregatorDecoLP(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregatorDecoLP, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids, unique_inverse_indices = np.unique(node_ids, return_inverse=True)
        unique_node_messages, unique_node_timestamps, to_update_node_ids, node_ids_inverse_index, to_not_update_node_ids = [], [], [], [], []

        for unique_idx, node_id in enumerate(unique_node_ids):
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(unique_idx)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])
            else:
                to_not_update_node_ids.append(unique_idx)
        # ndarray, shape (num_to_update_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)
        # ndarray, shape (num_to_not_update_unique_node_ids, ), array of unique node ids
        to_not_update_node_ids = np.array(to_update_node_ids)
        
        return unique_node_ids, to_update_node_ids, unique_node_messages, unique_node_timestamps, to_not_update_node_ids, unique_inverse_indices
