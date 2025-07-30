import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()

        self.covariate_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, cls_token):
        # inputs.shape [batch_size, num_atoms, num_timesteps, num_dims]
        B, N, T, D = x.shape

        # covariate transformer
        x = x.reshape(B, N * T, D)
        cls_token = cls_token.unsqueeze(1)
        x = torch.cat([x, cls_token], dim=1)
        x = self.covariate_transformer(x)
        cls_out = x[:, -1, :]
        x = x[:, :-1, :]
        x = x.reshape(B, N, T,D)

        return x, cls_out

import torch
from model.modules import *
from model.Encoder import Encoder
from torch.nn.functional import gumbel_softmax


class TransformerEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, factor)

        #self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

        self.in_proj = nn.Linear(n_in, n_hid)
        # Make positional encoding dynamic based on args
        self.num_atoms = args.num_atoms
        self.timesteps = args.timesteps
        self.pe = nn.Parameter(torch.randn(1, self.num_atoms, self.timesteps, n_hid) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, n_hid))

       # self.cross_transformer_0 = CrossTransformerLayer(d_model=n_hid, nhead=4, dim_feedforward=256)
        self.cross_transformer_1 = CrossTransformerLayer(d_model=n_hid, nhead=4, dim_feedforward=256)
        self.cross_transformer_2 = CrossTransformerLayer(d_model=n_hid, nhead=4, dim_feedforward=256)

        seq_len = self.timesteps
        self.next_change_index_offsetlayer = nn.Linear(n_hid, seq_len + 1)
        # last logit is for no change

         # N causal changes + 1 no change
        self.num_causal_changes = args.dynamic + 1


    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]

        x = self.in_proj(inputs)
        B, N, T, D = x.shape

        x = x + self.pe

        causal_change_index_mask = torch.zeros(B, T + 1, dtype=torch.float32, device=x.device)
        max_causal_change_count = self.num_causal_changes 
        transformer_output = x
        cls_out = self.cls_token.expand(B, -1)

        causal_graphs = []
        change_indicator_list = []       
        #transformer_output, cls_out = self.cross_transformer_0(transformer_output, cls_out)

        for i in range(max_causal_change_count):
            transformer_output, cls_out = self.cross_transformer_1(transformer_output, cls_out)
            transformer_output, cls_out = self.cross_transformer_2(transformer_output, cls_out)

            # from cls out predict the next change index
            next_causal_change_logits = self.next_change_index_offsetlayer(cls_out)
            if i == max_causal_change_count - 1:
                # force no change for the last causal change
                causal_change_index_mask[:, :-1] = -1e9
            next_causal_change_logits += causal_change_index_mask
            #next_causal_change_logits[:, -1] += 4  # bias for no change
            next_causal_change = gumbel_softmax(next_causal_change_logits, tau=1.0, hard=True)
            change_indicator_list.append(next_causal_change[:, :-1])

            # update the causal change index mask
            # to ensure that the next change index larger than the current change index
            causal_change_index_mask = (
                next_causal_change.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) * -1e9
            )
            # we always allow no change (last logit)
            causal_change_index_mask[:, -1] = 0

            # x.shape = [batch_size, num_atoms, num_timesteps, num_dims]
            x = transformer_output.mean(dim=2)
            # New shape: [num_sims, num_atoms, num_dims]

            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x_skip = x

            if self.factor:
                x = self.edge2node(x, rel_rec, rel_send)
                x = self.mlp3(x)
                x = self.node2edge(x, rel_rec, rel_send)
                x = torch.cat((x, x_skip), dim=2)  # Skip connection
                x = self.mlp4(x)
            else:
                x = self.mlp3(x)
                x = torch.cat((x, x_skip), dim=2)  # Skip connection
                x = self.mlp4(x)

            x = self.fc_out(x)

            causal_graphs.append(x.unsqueeze(-1))

        #   causal_graphs           :  (B, E, 2, C)
        causal_graphs = torch.cat(causal_graphs, dim=-1)

        # one‑hot change indicators
        phi = torch.stack(change_indicator_list, dim=1)            # (B, C, T)
        B, C, T = phi.shape

        # cumulative "step function":  step_g[t] = 1  ⇔  change g has happened by t
        step = torch.cumsum(phi, dim=-1)                           # (B, C, T)

        # build the segment masks
        remain = torch.ones(B, T, device=phi.device, dtype=phi.dtype)
        segment_masks = []                                         # list length C
        for g in range(C):
            if g < C - 1:
                # graph g lives UNTIL its own change happens
                mask_g = remain * (1.0 - step[:, g])               # (B, T)
                remain = remain * step[:, g]                       # what is left for the next graphs
            else:
                # last graph owns the rest
                mask_g = remain
            segment_masks.append(mask_g)

        assignment_weights = torch.stack(segment_masks, dim=-1)    # (B, T, C)

        # sanity check – every time‑step should be assigned to exactly one graph
        assert torch.allclose(assignment_weights.sum(-1), torch.ones_like(remain))

        graphs = torch.einsum('b e f c, b t c -> b t e f',
                            causal_graphs,
                            assignment_weights)                   # (B, T, E, F)
        return graphs
    
