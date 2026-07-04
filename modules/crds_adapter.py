"""Legacy downstream CRDS adapter implementation.

This module is kept only for compatibility with the earlier downstream-adapter
experiment. The main CRDS path now lives in SimCSE/IEM and should use
`modules.crds_utils` instead.
"""

import torch
import torch.nn as nn


class CRDSAdapter(nn.Module):
    def __init__(self, d_sem: int, d_cf: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        gate_hidden = max(d_model, min(1024, (d_sem + d_cf + 1) // 2))
        self.sem_proj = nn.Linear(d_sem, d_model)
        self.cf_proj = nn.Linear(d_cf, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_sem + d_cf + 1, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        self.input_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sem_emb: torch.Tensor,
        cf_emb: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if reliability.dim() == sem_emb.dim() - 1:
            reliability = reliability.unsqueeze(-1)
        reliability = reliability.to(dtype=sem_emb.dtype)
        gate_inputs = torch.cat(
            [self.input_dropout(sem_emb), self.input_dropout(cf_emb), reliability],
            dim=-1,
        )
        sem_z = self.sem_proj(self.input_dropout(sem_emb))
        cf_z = self.cf_proj(self.input_dropout(cf_emb))
        gate = torch.sigmoid(self.gate(gate_inputs))
        return self.layer_norm(gate * cf_z + (1.0 - gate) * sem_z)
