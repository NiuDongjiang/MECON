import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum

class NHGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=2, num_node_type=13,
                 use_far=True, far_dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_node_type = num_node_type
        self.use_far = bool(use_far)

        self.node_MLP = nn.ModuleDict({str(i): nn.Linear(hidden_dim, hidden_dim)
                                       for i in range(self.num_node_type)})

        self.score_MLP = nn.Linear(hidden_dim, num_heads, bias=False)
        self.edge_MLP  = nn.Linear(hidden_dim, hidden_dim)
        self.near_norm = nn.LayerNorm(hidden_dim)

        if self.use_far:
            self.tofs_lift = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(far_dropout),
            )
            self.type_emb = nn.Embedding(num_node_type, hidden_dim)
            self.tifo = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(far_dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.ttfi_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.fuse_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.far_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _default_batch(x):
        return torch.zeros((x.size(0),), device=x.device, dtype=torch.long)

    def _typed_linear_inplace(self, x, node_type_index):
        for t in range(self.num_node_type):
            m = (node_type_index == t)
            if m.any():
                x[m] = self.node_MLP[str(t)](x[m])
        return x

    def _near_field(self, x, edge_index, edge_attr):
        row, col = edge_index
        H = self.num_heads
        d = self.hidden_dim // H
        N = x.size(0)

        x_in = x
        x_h = x_in.view(-1, H, d)

        score_in = x_in[row] + x_in[col] + edge_attr
        score = self.score_MLP(score_in).view(-1, H, 1)
        score = F.leaky_relu(score, 0.2)

        alpha = scatter_softmax(src=score, index=row, dim=0)  # [E,H,1]
        msg = alpha * x_h[col]                                  # [E,H,d]
        agg = scatter_sum(src=msg, index=row, dim=0).reshape(N, -1)

        x_out = self.near_norm(agg + x_in)
        edge_attr_out = self.edge_MLP(x_out[row] + x_out[col] + edge_attr)
        return x_out, edge_attr_out

    def _build_type_near_mask(self, edge_index, node_type_index, num_types):
        row, col = edge_index
        tr = node_type_index[row]
        tc = node_type_index[col]
        idx = torch.cat([tr * num_types + tc, tc * num_types + tr], dim=0)

        A = torch.zeros((num_types * num_types,), device=edge_index.device, dtype=torch.bool)
        A[idx] = True
        A = A.view(num_types, num_types)
        A.fill_diagonal_(True)
        return A

    def _tofs_pool(self, x, node_type_index, batch):
        N, D = x.shape
        B = int(batch.max().item()) + 1
        T = self.num_node_type

        x_l = self.tofs_lift(x)
        w = torch.norm(x_l, p=2, dim=-1, keepdim=True).clamp_min(1e-8)  # [N,1]

        gid = batch * T + node_type_index  # [N]
        sum_wx = scatter_sum(w * x_l, gid, dim=0, dim_size=B * T)        # [B*T,D]
        sum_w  = scatter_sum(w, gid, dim=0, dim_size=B * T).clamp_min(1e-8)  # [B*T,1]
        q = (sum_wx / sum_w).view(B, T, D)
        cnt = sum_w.view(B, T, 1)
        return q, cnt

    def _tifo_far_translate(self, q, type_near, present_mask):
        B, T, D = q.shape
        e = self.type_emb.weight  # [T,D]
        e_t = e.view(1, T, 1, D).expand(B, T, T, D)
        e_s = e.view(1, 1, T, D).expand(B, T, T, D)
        q_s = q.view(B, 1, T, D).expand(B, T, T, D)

        far = (~type_near).view(1, T, T).expand(B, T, T)
        src_present = present_mask.view(B, 1, T).expand(B, T, T)
        far = far & src_present

        inp = torch.cat([q_s, e_t, e_s, (e_t - e_s)], dim=-1)
        trans = self.tifo(inp)
        trans = trans * far.unsqueeze(-1).float()
        h = trans.sum(dim=2)  # [B,T,D]
        return h

    def _ttfi_pushdown(self, x, node_type_index, batch, h):
        N, D = x.shape
        T = self.num_node_type

        h_i = h[batch, node_type_index]  # [N,D]

        dist2 = ((x - h_i) ** 2).sum(dim=-1, keepdim=True)       # [N,1]
        w = 1.0 / (1.0 + dist2)                                  # (0,1]
        w = w.clamp(max=1.0)

        gid = batch * T + node_type_index
        sum_w = scatter_sum(w, gid, dim=0, dim_size=(int(batch.max().item()) + 1) * T).clamp_min(1e-6)  # [B*T,1]
        w_norm = w / sum_w[gid]                                  # [N,1]

        x_pull = x + w_norm * (h_i - x)
        x_pull = self.ttfi_proj(x_pull)
        return x_pull

    def forward(self, x, edge_index, edge_attr, node_type_index, batch=None):
        if batch is None:
            batch = self._default_batch(x)

        x = self._typed_linear_inplace(x, node_type_index)
        x_near, edge_attr = self._near_field(x, edge_index, edge_attr)

        if not self.use_far:
            return x_near, edge_attr

        type_near = self._build_type_near_mask(edge_index, node_type_index, self.num_node_type)
        q, present_weight = self._tofs_pool(x_near, node_type_index, batch)
        present_mask = (present_weight.squeeze(-1) > 0)

        h = self._tifo_far_translate(q, type_near, present_mask)
        x_far = self._ttfi_pushdown(x_near, node_type_index, batch, h)
        x_far = self.far_norm(x_far)

        g = self.fuse_gate(torch.cat([x_near, x_far], dim=-1))
        x_out = (1 - g) * x_near + g * (x_near + x_far)
        return x_out, edge_attr

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HHGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_node_type=13, K=3, dropout=0.1, per_graph_coeff=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_node_type = num_node_type
        self.K = int(K)
        self.per_graph_coeff = bool(per_graph_coeff)

        self.node_linear_dict = nn.ModuleDict({
            str(i): nn.Linear(hidden_dim, hidden_dim) for i in range(self.num_node_type)
        })
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.out_norm  = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        if self.per_graph_coeff:
            self.coeff_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.K + 1)
            )
        else:
            self.coeff = nn.Parameter(torch.zeros(self.K + 1))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        fac = [math.factorial(k) for k in range(self.K + 1)]
        self.register_buffer("fac", torch.tensor(fac, dtype=torch.float).view(1, 1, 1, self.K + 1))

    def _typed_node_linear_in_dense_(self, embeds, node_type_index, node_mask):
        flat = embeds[node_mask]  # [N_total, D]
        for t in range(self.num_node_type):
            m = (node_type_index == t)
            if m.any():
                flat[m] = self.node_linear_dict[str(t)](flat[m])
        out = embeds.clone()
        out[node_mask] = flat
        return out

    @staticmethod
    def _graph_pool_mean(embeds, node_mask):
        m = node_mask.float().unsqueeze(-1)
        s = (embeds * m).sum(dim=1)
        c = m.sum(dim=1).clamp_min(1.0)
        return s / c

    @staticmethod
    def _normalize_adj(adj, node_mask, eps=1e-8):
        # adj: [B,N,H]
        adj = adj.float()

        # softmax over hyperedge dimension H
        adj = F.softmax(adj, dim=-1)  # non-negative, sum_H=1

        # mask padding nodes
        if node_mask is not None:
            adj = adj * node_mask.float().unsqueeze(-1)

        # 防止出现全 0 行导致后续 0/0：加 eps 并重新归一化
        row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
        adj = adj / row_sum
        return adj

    @staticmethod
    def _S(adj, X, eps=1e-6):
        """
        Normalized hypergraph propagation on non-negative adj:
          lat = D_e^{-1} adj^T X
          out = D_v^{-1} adj lat
        adj: [B,N,H] (non-negative normalized)
        X  : [B,N,D]
        """
        adj_t = adj.permute(0, 2, 1)  # [B,H,N]

        deg_e = adj_t.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B,H,1]
        lat = torch.bmm(adj_t, X) / deg_e                        # [B,H,D]

        deg_v = adj.sum(dim=-1, keepdim=True).clamp_min(eps)     # [B,N,1]
        out = torch.bmm(adj, lat) / deg_v                        # [B,N,D]
        return out

    def forward(self, adj, embeds, node_type_index, node_mask):
        embeds = self._typed_node_linear_in_dense_(embeds, node_type_index, node_mask)
        embeds = self.node_norm(embeds)

        B, N, D = embeds.shape

        adj = self._normalize_adj(adj, node_mask=node_mask)  # [B,N,H]

        if self.per_graph_coeff:
            g = self._graph_pool_mean(embeds, node_mask)     # [B,D]
            c = self.coeff_net(g)                            # [B,K+1]
        else:
            c = self.coeff.view(1, -1).expand(B, -1)

        c = F.softmax(c, dim=-1)  # 稳健优先

        basis = [embeds]
        X = embeds
        for _ in range(1, self.K + 1):
            X = self._S(adj, X)
            basis.append(X)

        Z = torch.stack(basis, dim=-1)                       # [B,N,D,K+1]
        ck = c.view(B, 1, 1, self.K + 1)
        scaled = ck / self.fac.to(embeds.dtype)
        out = (Z * scaled).sum(dim=-1)                       # [B,N,D]

        out = self.drop(self.proj(out))
        out = self.out_norm(out + embeds)
        return out
