from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_agents.layers import MLPLayer
from sim_agents.utils import weight_init


class MLPDecoder(nn.Module):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_prediction_steps: int,
        num_freq_bands: int,
        anchor_free: bool,
        num_modes: int,
    ) -> None:
        super(MLPDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_prediction_steps = num_prediction_steps
        self.num_freq_bands = num_freq_bands
        self.anchor_free = anchor_free
        self.num_modes = num_modes

        if self.anchor_free:
            self.mode_emb = nn.Embedding(num_modes, hidden_dim)
            self.to_pi = MLPLayer(input_dim=hidden_dim*2, hidden_dim=hidden_dim, output_dim=1)
        else:
            self.traj_emb = MLPLayer(
                input_dim=num_prediction_steps * (output_dim+1),
                hidden_dim=hidden_dim, output_dim=hidden_dim
            )

        self.to_loc_refine_pos = MLPLayer(
            input_dim=hidden_dim*2, hidden_dim=hidden_dim,
            output_dim=num_prediction_steps * output_dim
        )
        self.to_scale_refine_pos = MLPLayer(
            input_dim=hidden_dim*2, hidden_dim=hidden_dim,
            output_dim=num_prediction_steps * output_dim
        )
        self.to_loc_refine_head = MLPLayer(
            input_dim=hidden_dim*2, hidden_dim=hidden_dim,
            output_dim=num_prediction_steps * 2
        )
        self.to_conc_refine_head = MLPLayer(
            input_dim=hidden_dim*2, hidden_dim=hidden_dim,
            output_dim=num_prediction_steps
        )

        self.apply(weight_init)

    def forward(
        self,
        x_a: torch.Tensor,
        initial_traj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_agents = x_a.size(0)
        num_frames = x_a.size(1)
        assert x_a.size(2) == self.hidden_dim

        if self.anchor_free:
            num_modes = self.mode_emb.num_embeddings
            m = torch.cat([
                self.mode_emb.weight[None, None].expand(num_agents, num_frames, -1, -1), # (num_agents, num_frames, num_modes, hidden_dim)
                x_a.unsqueeze(2).expand(-1, -1, num_modes, -1) # (num_agents, num_frames, num_modes, hidden_dim)
            ], dim=-1) # (num_agents, num_frames, num_modes, hidden_dim * 2)

            # decode to trajectories with MLP
            loc_refine_pos = self.to_loc_refine_pos(m).view(
                num_agents, num_frames, num_modes, self.num_prediction_steps, self.output_dim
            )
            scale_refine_pos = F.elu_(self.to_scale_refine_pos(m).view(
                num_agents, num_frames, num_modes, self.num_prediction_steps, self.output_dim
            ), alpha=1.0) + 1.0 + 0.1

            loc_refine_head = self.to_loc_refine_head(m).view(
                num_agents, num_frames, num_modes, self.num_prediction_steps, 2
            )
            loc_refine_head = torch.atan2(loc_refine_head[..., 1], loc_refine_head[..., 0]).unsqueeze(-1)
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)

            # decode to scores with MLP
            pi = self.to_pi(m).squeeze(-1) # (num_agents, num_frames, num_modes)

            return {
                'loc_refine_pos': loc_refine_pos,
                'scale_refine_pos': scale_refine_pos,
                'loc_refine_head': loc_refine_head,
                'conc_refine_head': conc_refine_head,
                'pi': pi,
            }
        
        # embed propose trajectory
        assert initial_traj is not None
        m = self.traj_emb(initial_traj.view(num_agents, num_frames, -1)) # (num_agents, num_frames, hidden_dim)

        # concat propose trajectory embedding with agent embedding
        m = torch.cat([m, x_a], dim=-1) # (num_agents, num_frames, hidden_dim * 2)

        # decode to trajectory with MLP
        loc_refine_pos = self.to_loc_refine_pos(m).view(num_agents, num_frames, self.num_prediction_steps, self.output_dim)
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(num_agents, num_frames, self.num_prediction_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        
        loc_refine_head = self.to_loc_refine_head(m).view(num_agents, num_frames, self.num_prediction_steps, 2)
        loc_refine_head = torch.atan2(loc_refine_head[..., 1], loc_refine_head[..., 0]).unsqueeze(-1)
        conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)

        return {
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
        }
