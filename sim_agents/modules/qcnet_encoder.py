"""
Based on https://github.com/ZikangZhou/QCNet/blob/main/modules/qcnet_encoder.py
"""


from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from sim_agents.modules.qcnet_agent_encoder import QCNetAgentEncoder
from sim_agents.modules.qcnet_map_encoder import QCNetMapEncoder


class QCNetEncoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        key_frame_interval: int,
        time_span: int,
        pl2pl_knn: int,
        pl2a_knn: int,
        a2a_knn: int,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pl2pl_knn=pl2pl_knn,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            key_frame_interval=key_frame_interval,
            time_span=time_span,
            pl2a_knn=pl2a_knn,
            a2a_knn=a2a_knn,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(
        self,
        data: HeteroData,
        key_frames: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc, key_frames)
        return {**map_enc, **agent_enc}
