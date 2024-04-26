import numpy as np
import torch
import torch.nn as nn
import math

from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention, TT_DICT, AttentionFusion

from utils.utils import NeighborSampler, get_latest_unique_indices, vectorized_update_mem_2d
from models.MemoryModel import MemoryBank


class TGAT(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_nodes: int, use_init_method: bool, init_weights: str, time_partitioned_node_degrees: torch.Tensor, min_time: float, total_time:float, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        """
        TGAT model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        # * For init method * #
        self.memory_bank = MemoryBank(num_nodes = num_nodes, memory_dim = 1, device = self.device, k = 20)
        self.use_init_method = use_init_method
        self.init_weights = init_weights
        self.time_partitioned_node_degrees = time_partitioned_node_degrees
        self.min_time = min_time
        self.attfus = AttentionFusion(self.node_feat_dim)
        self.time_transformation_for_init = nn.ModuleList(
            [TT_DICT[name](min_time, total_time) for name in self.init_weights]
        )

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, edges_are_positive: bool, num_neighbors: int = 20, log_dict = None):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        
        all_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 0)
        all_node_ids = torch.cat(src_node_ids, dst_node_ids)
        updated_embeddings = self.get_init_node_memory_from_degree(node_ids=all_node_ids, node_memories=all_embeddings, node_interact_times=node_interact_times, log_dict=log_dict)
        src_node_embeddings, dst_node_embeddings = updated_embeddings[:src_node_embeddings.shape[0]], updated_embeddings[src_node_embeddings.shape[0]:]
        
        if edges_are_positive:
            src_unique_ids, src_latest_indices = get_latest_unique_indices(torch.from_numpy(src_node_ids))
            dst_unique_ids, dst_latest_indices = get_latest_unique_indices(torch.from_numpy(dst_node_ids))

            self.memory_bank.node_last_updated_times[src_unique_ids] = torch.from_numpy(node_interact_times)[src_latest_indices].float().to(self.device)
            self.memory_bank.node_last_updated_times[dst_unique_ids] = torch.from_numpy(node_interact_times)[dst_latest_indices].float().to(self.device)

            self.memory_bank.node_last_k_updated_times[src_unique_ids] = vectorized_update_mem_2d(self.memory_bank.node_last_k_updated_times[src_unique_ids], torch.from_numpy(node_interact_times)[src_latest_indices].float().to(self.device))
            self.memory_bank.node_last_k_updated_times[dst_unique_ids] = vectorized_update_mem_2d(self.memory_bank.node_last_k_updated_times[dst_unique_ids], torch.from_numpy(node_interact_times)[dst_latest_indices].float().to(self.device))

            self.memory_bank.is_node_seen[src_node_ids] = True
            self.memory_bank.is_node_seen[dst_node_ids] = True

            self.memory_bank.node_interact_counts = self.memory_bank.node_interact_counts.scatter_add(0, torch.from_numpy(src_node_ids).to(self.device), torch.ones_like(torch.from_numpy(src_node_ids), device = self.device))
            self.memory_bank.node_interact_counts = self.memory_bank.node_interact_counts.scatter_add(0, torch.from_numpy(dst_node_ids).to(self.device), torch.ones_like(torch.from_numpy(dst_node_ids), device = self.device))
            
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # Tensor, shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # Tensor, shape (batch_size, node_feat_dim)
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features
        else:
            # get source node representations by aggregating embeddings from the previous (current_layer_num - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)
            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors, self.node_feat_dim)

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_raw_features)

            return output

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def sample_nodes_acc_to_degree(self, num_samples, num_nodes_per_sample = 200, uniform = False):
        # Sample nodes
        probs = self.memory_bank.node_interact_counts ** 0.75
        samples = torch.multinomial(probs.reshape(1, -1).repeat(num_samples, 1), num_nodes_per_sample, replacement = False)
        return samples

    def get_init_node_memory_from_degree(self, node_ids, node_memories, node_interact_times, log_dict):
        """
        Updates the unseen nodes' embeddings to have a weighted average of embeddings of highly interacting nodes.
        node_ids: which node_ids are relevant
        node_memories: to update into and to calculate weighted average from
        """
        node_memories_ids = torch.from_numpy(node_memories_ids)
        node_ids = torch.from_numpy(node_ids).to(self.device)
        
        if self.time_partitioned_node_degrees is None:
            return node_memories_ids.cpu().detach().numpy(), node_memories
        
        node_memories_ids = node_memories_ids.to(self.device)
        weights = None
        # ****** GENERATE WEIGHTS FOR ALL NODES ******** #
        # If initialisation weight is degree or log degree
        weights = []
        for name, tt in zip(self.init_weights, self.time_transformation_for_init):
            if name == 'degree' or name == 'log-degree':
                num_partitions_total = self.time_partitioned_node_degrees.shape[0]
                check_time = float(torch.min(torch.from_numpy(node_interact_times)))
                partition_num = math.floor((check_time-self.min_time)*num_partitions_total/self.min_time) - 1
                weigh = self.time_partitioned_node_degrees[partition_num].clone()
                if name == 'log-degree':
                    weigh = torch.log(torch.max(torch.ones(1).to(weigh.device), weigh))
            else:
                last_k_times = self.memory_bank.node_last_k_updated_times
                curr_time = torch.max(torch.from_numpy(node_interact_times)).to(self.device)
                all_times = self.memory_bank.node_last_updated_times
                weigh = tt(last_k_times - curr_time, curr_time)
            weights.append(weigh)

        use_node_memories = node_memories
        # ****** GENERATE WEIGHTS FOR ALL NODES: DONE ******** #
                
        # ****** GENERATE NEW NODE EMBEDDINGS ******** #
        to_use_node_memories = use_node_memories.clone()
        if  weights is not None and torch.all(torch.tensor([torch.any(w != 0) for ws in weights for w in ws])):
            new_inits = []
            # CALCULATING SAMPLES EMBEDDINGS
            breakpoint()
            new_node_ids = (~self.memory_bank.is_node_seen[node_ids]).argwhere().reshape(-1)
            degree_based_overall_samples = self.sample_nodes_acc_to_degree(num_samples=1, num_nodes_per_sample=400).flatten()
            embeddings = self.compute_node_temporal_embeddings(node_ids=degree_based_overall_samples, node_interact_times=np.array(torch.ones_like(samples) * curr_time), current_layer_num=self.num_layers)
            
            # GENERATE MIXED SAMPLES FROM THESE SAMPLES FOR EACH NEW NODE
            if self.training:
                num_samples = 32
                num_nodes_per_sample = 200
            else:
                num_samples = 2
                num_nodes_per_sample = 400
            samples = torch.multinomial(torch.ones(num_samples, degree_based_overall_samples.shape[0]), num_nodes_per_sample, replacement = False)
            # shape (num_samples, num_nodes_per_sample)
            
            # CALCULATE NEW INIT EMBEDDINGS FOR MIXED SAMPLES USING ATTENTION FUSION + WEIGHTED INITIALISATION
            for weigh in weights:
                mems = embeddings[samples]
                ws = weigh[degree_based_overall_samples[samples]]
                new_node_init = (ws.unsqueeze(2) * mems).sum(dim = 1) / ws.sum(dim = 1).reshape(-1, 1)
                new_inits.append(new_node_init.unsqueeze(1))
            new_init_embeds = torch.cat(new_inits, dim = 1)
            # Run attention fusion operation
            new_init = self.attfus(new_init_embeds, log_dict)
            new_init_repeated = torch.zeros_like(to_use_node_memories)
            new_init = new_init[torch.randperm(new_node_ids.shape[0]) % num_samples]
            new_init_repeated[new_node_ids] += new_init
            
            # UPDATE THE EMBEDDINGS OF THE NEW NODES USING MASKING
            mask = torch.zeros(to_use_node_memories.shape[0]).to(self.device)
            mask[new_node_ids] = 1
            cloned = new_init_repeated.clone()
            use_node_memories = to_use_node_memories + (mask.unsqueeze(-1) * cloned)
            if log_dict:
                log_dict['new_init_mean'] = torch.mean(new_init_repeated.detach())
                log_dict['new_init_std'] = torch.std(new_init_repeated.detach())
            del new_init_repeated, cloned, mask, new_init, new_init_embeds, new_inits, samples, weights, to_use_node_memories
        else:
            return node_memories_ids.cpu().detach().numpy(), node_memories