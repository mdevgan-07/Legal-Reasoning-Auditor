"""
WeightedCRF — your training code's CRF, verbatim semantics (class-weighted
log-likelihood, Viterbi decode). Kept identical so results are attributable
to the changes we're testing, not silent CRF differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedCRF(nn.Module):
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx,
                 pad_tag_idx=None, class_weights=None):
        super().__init__()
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(n_tags))

        self.transitions = nn.Parameter(torch.empty(n_tags, n_tags))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.SOS_TAG_IDX] = -1e6
        self.transitions.data[self.EOS_TAG_IDX, :] = -1e6
        if self.PAD_TAG_IDX is not None:
            self.transitions.data[self.PAD_TAG_IDX, :] = -1e6
            self.transitions.data[:, self.PAD_TAG_IDX] = -1e6
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0

    def forward(self, emissions, tags, mask=None):
        return -self.log_likelihood(emissions, tags, mask=mask)

    def log_likelihood(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], device=emissions.device)
        scores = self._compute_scores(emissions, tags, mask)
        partition = self._compute_log_partition(emissions, mask)
        batch_size = tags.shape[0]
        weights = torch.zeros(batch_size, device=emissions.device)
        for i in range(batch_size):
            seq_len = mask[i].int().sum().item()
            seq_tags = tags[i, :seq_len]
            weights[i] = self.class_weights[seq_tags].mean()
        return torch.sum(weights * (scores - partition))

    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        device = emissions.device
        scores = torch.zeros(batch_size, device=device)
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
        scores += self.transitions[self.SOS_TAG_IDX, first_tags]
        scores += emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)
        for i in range(1, seq_len):
            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            e = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze(1)
            t = self.transitions[prev_tags, curr_tags]
            scores += e * is_valid + t * is_valid
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, _ = emissions.shape
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
        end_scores = alphas + self.transitions[:, self.EOS_TAG_IDX].unsqueeze(0)
        return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], device=emissions.device)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, _ = emissions.shape
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for i in range(1, seq_len):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            max_scores, max_tags = torch.max(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            backpointers.append(max_tags.t())
        end_scores = alphas + self.transitions[:, self.EOS_TAG_IDX].unsqueeze(0)
        _, max_final_tags = torch.max(end_scores, dim=1)
        best_sequences = []
        lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            length = lengths[i].item()
            final_tag = max_final_tags[i].item()
            bps = backpointers[: length - 1]
            path = self._find_best_path(i, final_tag, bps)
            best_sequences.append(path)
        return best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        best_path = [best_tag]
        for bps_t in reversed(backpointers):
            best_tag = bps_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)
        return best_path
