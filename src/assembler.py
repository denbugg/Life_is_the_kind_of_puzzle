"""End-to-end differentiable jigsaw assembler (Sinkhorn / DeepPermNet style).
Ingests the 576 fragments as an unordered SET of DINOv2 descriptors, reasons
globally via self-attention, and outputs a soft permutation (fragment -> grid cell)
through a differentiable Sinkhorn operator. Trained against the known permutation.
A radically different paradigm from pairwise-compatibility + search. See NEW_CONCEPT.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GRID, NFRAG


def log_sinkhorn(log_alpha, n_iter=20):
    """Sinkhorn normalisation in log-space -> log of a doubly-stochastic matrix."""
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha


class AssemblerNet(nn.Module):
    """Input: precomputed per-fragment features (B, N, feat_dim). No positional
    encoding on the input tokens -> permutation-equivariant. Output: log soft-perm."""
    def __init__(self, feat_dim=435, d=256, layers=6, heads=8, sink_iter=20, tau=0.5):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.frag_proj = nn.Linear(d, d)
        # learned cell queries with a 2-D grid positional prior
        rc = torch.stack(torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij"), -1)
        self.register_buffer("cell_rc", rc.reshape(NFRAG, 2).float() / (GRID - 1))
        self.cell_pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.cell_emb = nn.Parameter(torch.randn(NFRAG, d) * 0.02)
        self.sink_iter = sink_iter
        self.tau = tau
        self.scale = d ** -0.5

    def forward(self, feats):                          # feats: (B, N, feat_dim)
        e = self.tf(self.inp(feats))                   # (B, N, d) global context
        q = self.frag_proj(e)
        cells = self.cell_emb + self.cell_pos(self.cell_rc)          # (N, d)
        A = torch.einsum("bnd,md->bnm", q, cells) * self.scale       # (B, N, N) frag->cell
        logP = log_sinkhorn(A / self.tau, self.sink_iter)
        return logP, A


def assemble_loss(logP, target):
    """target: (B,N) true cell index per fed fragment. Symmetric row+col NLL."""
    B, N, _ = logP.shape
    row = F.nll_loss(logP.reshape(B * N, N), target.reshape(B * N))
    inv = torch.argsort(target, dim=1)             # inv[b, cell] = fragment placed there
    logPt = logP.transpose(1, 2)                   # (B, cell, frag)
    col = F.nll_loss(logPt.reshape(B * N, N), inv.reshape(B * N))
    return 0.5 * (row + col)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
