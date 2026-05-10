import torch
from torch import nn
import torch.nn.functional as F


class MoEGate(nn.Module):


    def __init__(self, in_dim, n_expert=2, top_k=2, dropout=0.1, temperature=1.0):
        super().__init__()
        self.n_expert = n_expert
        self.top_k = top_k
        self.temperature = temperature

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_expert)
        )

    def forward(self, gate_in):
        # gate_in: [B, in_dim]
        logits = self.mlp(gate_in) / self.temperature
        probs = F.softmax(logits, dim=-1)  # [B, 2]

        if self.top_k >= self.n_expert:
            mask = torch.ones_like(probs)
            routed = probs
        else:
            topk_val, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_val))
            routed = probs * mask
            routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-12)

        return routed, probs, mask


class MVN_DDI(nn.Module):


    def __init__(self, hidden_dim=128, ddi_type=86, drop=0.2, bal_beta=0.01, moe_topk=2):
        super().__init__()
        self.bal_beta = bal_beta
        self.hidd_dim = hidden_dim

        self.mol_ctx_proj = nn.Sequential(
            nn.Linear(self.hidd_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(drop)
        )

        self.kg_ctx_proj = nn.Sequential(
            nn.Linear(self.hidd_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(drop)
        )

        self.type_embedding = nn.Embedding(ddi_type, hidden_dim)

        self.gate = MoEGate(
            in_dim=hidden_dim * 3,   # mol_ctx + kg_ctx + rel_ctx
            n_expert=2,
            top_k=moe_topk,
            dropout=drop,
            temperature=1.0
        )

    def _balance_loss(self, probs, routed):

        mean_probs = probs.mean(dim=0)      # [2]
        mean_routed = routed.mean(dim=0)    # [2]

        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())

        loss_align = torch.sum((mean_routed - mean_probs) ** 2)
        loss_uniform = torch.sum((mean_routed - uniform) ** 2)

        return loss_align + loss_uniform

    def forward(self, score_mol, score_kg, h_pool, t_pool, KG_d1_feat, KG_d2_feat, DDI_type_index):
        # Context encoders
        mol_ctx = self.mol_ctx_proj(torch.cat([h_pool, t_pool], dim=-1))          # [B, H]
        kg_ctx = self.kg_ctx_proj(torch.cat([KG_d1_feat, KG_d2_feat], dim=-1))    # [B, H]
        rel_ctx = self.type_embedding(DDI_type_index)                              # [B, H]

        gate_in = torch.cat([mol_ctx, kg_ctx, rel_ctx], dim=-1)                   # [B, 3H]
        routed, probs, mask = self.gate(gate_in)                                   # [B, 2]

        mol_w = routed[:, 0]
        kg_w = routed[:, 1]

        mol_score_routed = mol_w * score_mol
        kg_score_routed = kg_w * score_kg
        fuse_score = mol_score_routed + kg_score_routed

        bal_loss = None
        if self.training:
            bal_loss = self.bal_beta * self._balance_loss(probs, routed)

        return {
            "fuse_score": fuse_score,                   # [B]
            "mol_score_routed": mol_score_routed,      # [B]
            "kg_score_routed": kg_score_routed,        # [B]
            "mol_weight": mol_w,                       # [B]
            "kg_weight": kg_w,                         # [B]
            "routed": routed,                          # [B, 2]
            "probs": probs,                            # [B, 2]
            "mask": mask,                              # [B, 2]
            "bal_loss": bal_loss
        }