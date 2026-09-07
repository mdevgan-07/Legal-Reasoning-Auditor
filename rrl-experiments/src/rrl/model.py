"""
Unified RRL model: (encoder | cached embeddings) -> sequence layer
(BiLSTM or Transformer) -> emissions -> WeightedCRF, with an optional
auxiliary focal cross-entropy on the emissions.

Focal note: a true focal-CRF is analytically messy; the standard, effective
alternative is  loss = CRF_NLL + λ · FocalCE(emissions, gold). The CRF keeps
structure; focal reweights hard/rare sentences (Issue, Decision) at the
emission level. λ=0 disables it exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crf import WeightedCRF
from .config import ExperimentConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                       # x: [B, N, D]
        return x + self.pe[:, : x.size(1)]


class RRLModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, n_tags: int, sos: int, eos: int,
                 pad: int, in_dim: int, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device_str = device
        self.n_tags = n_tags

        d = cfg.hidden_dim
        self.in_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()

        if cfg.sequence == "bilstm":
            self.seq = nn.LSTM(d, d // 2, bidirectional=True, batch_first=True)
            self.seq_drop = nn.Dropout(cfg.lstm_dropout)
            self._is_lstm = True
        elif cfg.sequence == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.tf_heads, dim_feedforward=cfg.tf_ff,
                dropout=cfg.tf_dropout, batch_first=True, norm_first=True,
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=cfg.tf_layers)
            self.pos = PositionalEncoding(d, max_len=max(cfg.hard_cap, cfg.max_sents) + 8) \
                if cfg.positional else None
            self.seq_drop = nn.Dropout(cfg.tf_dropout)
            self._is_lstm = False
        else:
            raise ValueError(f"unknown sequence layer: {cfg.sequence}")

        self.hidden2tag = nn.Linear(d, n_tags)
        cw = torch.tensor(cfg.class_weights, dtype=torch.float32)
        self.crf = WeightedCRF(n_tags, sos, eos, pad, cw)
        self.pad_idx = pad

        # aux None-vs-content binary head (targets Reasoning<->None confusion):
        # shares the sequence representation; trained jointly, unused at decode.
        self.none_head = nn.Linear(d, 1) if cfg.aux_none_lambda > 0 else None
        self.none_id: int | None = None            # set by trainer (tag2idx['None'])
        self._last_seq_out = None                  # cached for aux loss

        # focal CE class weights: reuse CRF class weights (0 for structural tags)
        self.register_buffer("focal_w", cw.clone())

    def set_class_weights(self, weights: list[float]) -> None:
        """Replace class weights post-construction (effective-number scheme)."""
        w = torch.tensor(weights, dtype=torch.float32, device=self.focal_w.device)
        self.focal_w.copy_(w)
        self.crf.class_weights.copy_(w)

    # ------------------------------------------------------------------
    def emissions_from_embeddings(self, doc_embs: list[torch.Tensor]):
        """doc_embs: list of [n_i, in_dim] tensors -> (emissions [B,N,T], mask)."""
        B = len(doc_embs)
        lens = [e.size(0) for e in doc_embs]
        N = max(lens)
        D = doc_embs[0].size(1)
        dev = doc_embs[0].device
        padded = torch.zeros(B, N, D, device=dev)
        mask = torch.zeros(B, N, device=dev)
        for i, e in enumerate(doc_embs):
            padded[i, : lens[i]] = e
            mask[i, : lens[i]] = 1.0

        x = self.in_proj(padded)
        if self._is_lstm:
            out, _ = self.seq(x)
        else:
            if self.pos is not None:
                x = self.pos(x)
            out = self.seq(x, src_key_padding_mask=(mask == 0))
        out = self.seq_drop(out)
        self._last_seq_out = out                    # cached for aux none head
        return self.hidden2tag(out), mask

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

    # ------------------------------------------------------------------
    def loss(self, emissions, mask, gold: list[list[int]]):
        tags = [torch.tensor(g, dtype=torch.long, device=emissions.device) for g in gold]
        tags = nn.utils.rnn.pad_sequence(tags, batch_first=True,
                                         padding_value=self.pad_idx)
        # pad tags tensor to emission length if needed
        if tags.size(1) < emissions.size(1):
            pad = torch.full((tags.size(0), emissions.size(1) - tags.size(1)),
                             self.pad_idx, dtype=torch.long, device=tags.device)
            tags = torch.cat([tags, pad], dim=1)

        crf_nll = self.crf(emissions, tags, mask=mask)
        total = crf_nll
        valid = mask.bool()

        lam = self.cfg.focal_lambda
        if lam > 0:
            logits = emissions[valid]                 # [M, T]
            target = tags[valid]                      # [M]
            logp = F.log_softmax(logits, dim=-1)
            eps = self.cfg.label_smoothing
            if eps > 0:
                # smoothed NLL: (1-eps)*logp_target + eps*mean(logp)
                nll_t = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
                nll_u = -logp.mean(dim=-1)
                nll = (1 - eps) * nll_t + eps * nll_u
            else:
                nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
            p_t = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
            w = self.focal_w[target]
            focal = (w * (1 - p_t) ** self.cfg.focal_gamma * nll)
            focal = focal.sum() / max(1.0, float(valid.sum()))
            total = total + lam * focal * mask.size(0)

        # aux None-vs-content head — binary CE against (gold == None)
        if self.none_head is not None and self.cfg.aux_none_lambda > 0 \
                and self.none_id is not None:
            seq = self._last_seq_out[valid]           # [M, d]
            logit = self.none_head(seq).squeeze(-1)   # [M]
            target_bin = (tags[valid] == self.none_id).float()
            bce = F.binary_cross_entropy_with_logits(logit, target_bin)
            total = total + self.cfg.aux_none_lambda * bce * mask.size(0)

        return total
