"""
Based on https://github.com/ZikangZhou/QCNet/blob/main/modules/qcnet_agent_encoder.py
"""


from typing import Dict, Mapping, Optional
import math

import torch
import torch.nn as nn
from torch_cluster import knn
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from sim_agents.layers.attention_layer import AttentionLayer
from sim_agents.layers.fourier_embedding import FourierEmbedding
from sim_agents.utils import angle_between_2d_vectors
from sim_agents.utils import weight_init
from sim_agents.utils import wrap_angle


class QCNetAgentEncoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        key_frame_interval: int,
        time_span: int,
        pl2a_knn: int,
        a2a_knn: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.key_frame_interval = key_frame_interval
        self.time_span = time_span
        self.pl2a_knn = pl2a_knn
        self.a2a_knn = a2a_knn
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'waymo':
            input_dim_x_a = 7
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3

            self.type_a_emb = nn.Embedding(5, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_key_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_key_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                               dropout=dropout, bipartite=False, has_pos_emb=True)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def encode_agent(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        ## read agent data
        mask_a = data['agent']['valid_mask'] # (num_agents, num_steps)
        num_agents, num_steps = mask_a.size()
        
        pos_a = data['agent']['position'][..., :self.input_dim] # (num_agents, num_steps, input_dim)
        motion_vector_a = torch.cat([
            pos_a.new_zeros(num_agents, 1, self.input_dim),
            pos_a[:, 1:] - pos_a[:, :-1]
        ], dim=1) # (num_agents, num_steps, input_dim) TODO: note `vector_repr` in dataset setting

        head_a = data['agent']['heading'] # (num_agents, num_steps)
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1) # (num_agents, num_steps, 2)

        if self.dataset == 'waymo':
            vel_a = data['agent']['velocity'][..., :self.input_dim] # (num_agents, num_steps, input_dim)

            length_a = data['agent']['length'].unsqueeze(1).repeat(1, num_steps) # (num_agents, num_steps)
            width_a = data['agent']['width'].unsqueeze(1).repeat(1, num_steps) # (num_agents, num_steps)
            height_a = data['agent']['height'].unsqueeze(1).repeat(1, num_steps) # (num_agents, num_steps)

            type_a = data['agent']['type'] # (num_agents,)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        ## encode agent trajectories
        if self.dataset == 'waymo':
            x_a = torch.stack([
                torch.norm(motion_vector_a[..., :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[..., :2]),
                torch.norm(vel_a[..., :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel_a[..., :2]),
                length_a, width_a, height_a
            ], dim=-1)
            categorical_embs = [
                self.type_a_emb(type_a.long()).repeat_interleave(repeats=num_steps, dim=0),
            ]
            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
            x_a = x_a.view(num_agents, num_steps, self.hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        return {
            'mask_a': mask_a,
            'pos_a': pos_a,
            'head_a': head_a,
            'head_vector_a': head_vector_a,
            'x_a': x_a
        }
    
    def temporal_compress(
        self,
        agent_enc: Dict[str, torch.Tensor],
        key_frames = None,
    ) -> Dict[str, torch.Tensor]:
        ## read agent data
        mask_a = agent_enc['mask_a']
        pos_a = agent_enc['pos_a']
        head_a = agent_enc['head_a']
        head_vector_a = agent_enc['head_vector_a']
        num_agents, num_steps = mask_a.size()

        ## key frame selection
        if key_frames is None:
            # ensure the last frame is a key frame
            start_frame = (num_steps % self.key_frame_interval) - 1
            if start_frame < 0:
                start_frame += self.key_frame_interval
            key_frames = torch.arange(start_frame, num_steps, self.key_frame_interval, device=mask_a.device)
        # sort key frames
        key_frames = key_frames.sort().values.long().unique()

        mask_key = mask_a[:, key_frames]
        pos_key = pos_a[:, key_frames]
        head_key = head_a[:, key_frames]
        head_vector_key = head_vector_a[:, key_frames]
        
        ## temporal mask
        mask_t_key = mask_a.unsqueeze(2) & mask_a.unsqueeze(1) # (num_agents, num_steps, num_steps)
        last_frame = 0
        mask_t_key[..., 0] = False
        for frame in key_frames:
            mask_t_key[..., last_frame+1:frame] = False
            mask_t_key[:, :last_frame, frame] = False
            mask_t_key[:, frame:, frame] = False
            last_frame = frame
        edge_index_t_key = dense_to_sparse(mask_t_key)[0] # [all_frames, key_frames]

        ## relative pose
        pos_a = pos_a.reshape(-1, self.input_dim)
        head_a = head_a.reshape(-1)
        head_vector_a = head_vector_a.reshape(-1, 2)
        rel_pos_t_key = pos_a[edge_index_t_key[0]] - pos_a[edge_index_t_key[1]]
        rel_head_t_key = wrap_angle(head_a[edge_index_t_key[0]] - head_a[edge_index_t_key[1]])
        r_t_key = torch.stack([
            torch.norm(rel_pos_t_key[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_t_key[1]], nbr_vector=rel_pos_t_key[:, :2]),
            rel_head_t_key,
            edge_index_t_key[0] - edge_index_t_key[1]
        ], dim=-1)
        r_t_key = self.r_t_key_emb(continuous_inputs=r_t_key, categorical_embs=None)
        
        ## temporal attention
        x_a = agent_enc['x_a'].reshape(-1, self.hidden_dim)
        x_a = self.t_key_attn_layer(x_a, r_t_key, edge_index_t_key)
        x_a = x_a.view(num_agents, num_steps, self.hidden_dim)
        x_key = x_a[:, key_frames]

        return {
            'mask_a': mask_key,
            'pos_a': pos_key,
            'head_a': head_key,
            'head_vector_a': head_vector_key,
            'x_a': x_key
        }

    def forward(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        key_frames: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        ## encode agent data
        agent_enc = self.encode_agent(data)

        ## temporal compress
        agent_enc = self.temporal_compress(agent_enc, key_frames)
        mask_a = agent_enc['mask_a']
        pos_a = agent_enc['pos_a']
        head_a = agent_enc['head_a']
        head_vector_a = agent_enc['head_vector_a']
        x_a = agent_enc['x_a']
        num_agents, num_steps = mask_a.size()

        ## temporal relations
        pos_a = pos_a.reshape(-1, self.input_dim) # (num_agents * num_steps, input_dim)
        head_a = head_a.reshape(-1) # (num_agents * num_steps,)
        head_vector_a = head_vector_a.reshape(-1, 2) # (num_agents * num_steps, 2)

        mask_t = mask_a.unsqueeze(2) & mask_a.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] < math.ceil(self.time_span / self.key_frame_interval)
        ]
        
        rel_pos_t = pos_a[edge_index_t[0]] - pos_a[edge_index_t[1]]
        rel_head_t = wrap_angle(head_a[edge_index_t[0]] - head_a[edge_index_t[1]])
        r_t = torch.stack([
            torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
            rel_head_t,
            edge_index_t[0] - edge_index_t[1]
        ], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        ## map to agent relations
        pos_pl = data['map_polygon']['position'][:, :self.input_dim] # (num_map_polygons, input_dim)
        orient_pl = data['map_polygon']['orientation'] # (num_map_polygons,)
        mask_a = mask_a.reshape(-1) # (num_agents * num_steps,)
        
        index_a, index_pl = knn(
            x=pos_pl[:, :2], y=pos_a[:, :2],
            k=self.pl2a_knn,
            batch_x=data['map_polygon']['batch'],
            batch_y=data['agent']['batch'].repeat_interleave(num_steps)
        )
        edge_index_pl2a = torch.stack([index_pl, index_a], dim=0)
        edge_index_pl2a = edge_index_pl2a[:, mask_a[edge_index_pl2a[1]]]

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_a[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_a[edge_index_pl2a[1]])
        r_pl2a = torch.stack([
            torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
            rel_orient_pl2a
        ], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        ## agent to agent relations
        pos_a = pos_a.reshape(num_agents, num_steps, self.input_dim).transpose(0, 1).reshape(-1, self.input_dim)
        head_a = head_a.reshape(num_agents, num_steps).transpose(0, 1).reshape(-1)
        head_vector_a = head_vector_a.reshape(num_agents, num_steps, 2).transpose(0, 1).reshape(-1, 2)
        mask_a = mask_a.reshape(num_agents, num_steps).transpose(0, 1).reshape(-1)

        batch_a2a = torch.cat([
            data['agent']['batch'] + data.num_graphs * t for t in range(num_steps)
        ], dim=0)

        edge_index_a2a = knn_graph(
            x=pos_a[:, :2],
            k=self.a2a_knn,
            batch=batch_a2a,
            loop=False
        )
        edge_index_a2a = subgraph(subset=mask_a, edge_index=edge_index_a2a)[0]

        rel_pos_a2a = pos_a[edge_index_a2a[0]] - pos_a[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_a[edge_index_a2a[0]] - head_a[edge_index_a2a[1]])
        r_a2a = torch.stack([
            torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
            rel_head_a2a
        ], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        ## Attention layers
        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim) # (num_agents * num_steps, hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'], x_a), r_pl2a, edge_index_pl2a)
            x_a = x_a.reshape(num_agents, num_steps, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(num_steps, num_agents, self.hidden_dim).transpose(0, 1) # (num_agents, num_steps, hidden_dim)

        return {
            'x_a': x_a
        }
    
    def inference(
        self,
        data: HeteroData,
        map_enc: Mapping[str, torch.Tensor],
        key_frames: Optional[torch.Tensor] = None,
        temporal_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        
        ## encode agent data
        agent_enc = self.encode_agent(data)
        update_x_a_origin = agent_enc['x_a'][:, -1]
        if temporal_cache is not None:
            x_a_origin = temporal_cache['x_a_origin']
            x_a_update = agent_enc['x_a'][:, 1:]
            agent_enc['x_a'] = torch.cat([x_a_origin.unsqueeze(1), x_a_update], dim=1)
        
        ## temporal compress
        agent_enc = self.temporal_compress(agent_enc, key_frames)
        mask_a = agent_enc['mask_a']
        pos_a = agent_enc['pos_a']
        head_a = agent_enc['head_a']
        head_vector_a = agent_enc['head_vector_a']
        x_a = agent_enc['x_a']
        num_agents, num_steps = mask_a.size()

        ## temporal relations
        if temporal_cache is not None:
            mask_a_cache = temporal_cache['mask_a']
            pos_a_cache = temporal_cache['pos_a']
            head_a_cache = temporal_cache['head_a']
            head_vector_a_cache = temporal_cache['head_vector_a']
            num_cache_agents, num_cache_steps = mask_a_cache.size()
            assert num_agents == num_cache_agents

            mask_a_all = torch.cat([mask_a_cache, mask_a], dim=1)
            pos_a_all = torch.cat([pos_a_cache, pos_a], dim=1)
            head_a_all = torch.cat([head_a_cache, head_a], dim=1)
            head_vector_a_all = torch.cat([head_vector_a_cache, head_vector_a], dim=1)

            mask_t = mask_a_all.unsqueeze(2) & mask_a_all.unsqueeze(1)
            mask_t[:, :, :num_cache_steps] = False
        else:
            mask_a_all = mask_a
            pos_a_all = pos_a
            head_a_all = head_a
            head_vector_a_all = head_vector_a

            mask_t = mask_a_all.unsqueeze(2) & mask_a_all.unsqueeze(1)
        
        pos_t = pos_a_all.reshape(-1, self.input_dim)
        head_t = head_a_all.reshape(-1)
        head_vector_t = head_vector_a_all.reshape(-1, 2)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] < math.ceil(self.time_span / self.key_frame_interval)
        ]
        
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack([
            torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
            rel_head_t,
            edge_index_t[0] - edge_index_t[1]
        ], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        ## map to agent relations
        pos_pl = data['map_polygon']['position'][:, :self.input_dim] # (num_map_polygons, input_dim)
        orient_pl = data['map_polygon']['orientation'] # (num_map_polygons,)
        mask_a = mask_a.reshape(-1) # (num_agents * num_steps,)
        pos_a = pos_a.reshape(-1, self.input_dim) # (num_agents * num_steps, input_dim)
        head_a = head_a.reshape(-1) # (num_agents * num_steps,)
        head_vector_a = head_vector_a.reshape(-1, 2) # (num_agents * num_steps, 2)
        
        index_a, index_pl = knn(
            x=pos_pl[:, :2], y=pos_a[:, :2],
            k=self.pl2a_knn,
            batch_x=data['map_polygon']['batch'],
            batch_y=data['agent']['batch'].repeat_interleave(num_steps)
        )
        edge_index_pl2a = torch.stack([index_pl, index_a], dim=0)
        edge_index_pl2a = edge_index_pl2a[:, mask_a[edge_index_pl2a[1]]]

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_a[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_a[edge_index_pl2a[1]])
        r_pl2a = torch.stack([
            torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
            rel_orient_pl2a
        ], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        ## agent to agent relations
        pos_a = pos_a.reshape(num_agents, num_steps, self.input_dim).transpose(0, 1).reshape(-1, self.input_dim)
        head_a = head_a.reshape(num_agents, num_steps).transpose(0, 1).reshape(-1)
        head_vector_a = head_vector_a.reshape(num_agents, num_steps, 2).transpose(0, 1).reshape(-1, 2)
        mask_a = mask_a.reshape(num_agents, num_steps).transpose(0, 1).reshape(-1)

        batch_a2a = torch.cat([
            data['agent']['batch'] + data.num_graphs * t for t in range(num_steps)
        ], dim=0)

        edge_index_a2a = knn_graph(
            x=pos_a[:, :2],
            k=self.a2a_knn,
            batch=batch_a2a,
            loop=False
        )
        edge_index_a2a = subgraph(subset=mask_a, edge_index=edge_index_a2a)[0]

        rel_pos_a2a = pos_a[edge_index_a2a[0]] - pos_a[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_a[edge_index_a2a[0]] - head_a[edge_index_a2a[1]])
        r_a2a = torch.stack([
            torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
            rel_head_a2a
        ], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        ## Attention layers
        x_a_all = {}
        for i in range(self.num_layers):
            if temporal_cache is not None:
                x_a_cache_i = temporal_cache['x_a'][i] # (num_cache_agents, num_cache_steps, hidden_dim)
                x_a_all_i = torch.cat([x_a_cache_i, x_a], dim=1)
            else:
                x_a_all_i = x_a
            x_a_all[i] = x_a_all_i

            x_t = x_a_all_i.reshape(-1, self.hidden_dim)
            x_t = self.t_attn_layers[i](x_t, r_t, edge_index_t)
            x_a = x_t.reshape(num_agents, -1, self.hidden_dim)[:, -num_steps:]

            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'], x_a), r_pl2a, edge_index_pl2a)

            x_a = x_a.reshape(num_agents, num_steps, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(num_steps, num_agents, self.hidden_dim).transpose(0, 1) # (num_agents, num_steps, hidden_dim)
        
        ## update temporal cache
        time_span_steps = math.ceil(self.time_span / self.key_frame_interval)
        update_temporal_cache = {
            'x_a_origin': update_x_a_origin,
            'mask_a': mask_a_all[:, -time_span_steps:],
            'pos_a': pos_a_all[:, -time_span_steps:],
            'head_a': head_a_all[:, -time_span_steps:],
            'head_vector_a': head_vector_a_all[:, -time_span_steps:],
            'x_a': {l: x_a_all[l][:, -time_span_steps:] for l in range(self.num_layers)}
        }

        return {
            'x_a': x_a,
            'temporal_cache': update_temporal_cache
        }
