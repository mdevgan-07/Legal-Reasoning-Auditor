"""
Sentence encoder: InLegalBERT with configurable pooling, fine-tune depth,
optional LoRA (Tier C), and a frozen mode (Tier B two-stage).

Pooling options:
  cls       — [CLS] token (your old setup)
  mean      — masked mean over tokens (uses full sentence information)
  cls_mean  — concat of both -> 1536-dim (sequence layer input adapts)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):
    def __init__(self, bert_dir: str, pooling: str = "cls_mean",
                 n_fine_tune_layers: int = 12, bert_batch_size: int = 16,
                 lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, freeze: bool = False,
                 grad_checkpoint: bool = False,
                 device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel

        self.pooling = pooling
        self.bert_batch_size = bert_batch_size
        self.device_str = device
        self.bert = AutoModel.from_pretrained(bert_dir)

        if lora:
            # Tier C: LoRA adapters on attention projections; base frozen.
            from peft import LoraConfig, get_peft_model
            for p in self.bert.parameters():
                p.requires_grad = False
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense"],
                bias="none",
            )
            self.bert = get_peft_model(self.bert, peft_cfg)
        elif freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = False
            n_layers = self.bert.config.num_hidden_layers
            n_ft = min(n_fine_tune_layers, n_layers)
            for i in range(n_layers - n_ft, n_layers):
                for p in self.bert.encoder.layer[i].parameters():
                    p.requires_grad = True
            if getattr(self.bert, "pooler", None) is not None:
                for p in self.bert.pooler.parameters():
                    p.requires_grad = True
        if grad_checkpoint and not freeze and not lora:
            self.bert.gradient_checkpointing_enable()
            self.bert.enable_input_require_grads()

    @property
    def out_dim(self) -> int:
        h = self.bert.config.hidden_size
        return 2 * h if self.pooling == "cls_mean" else h

    def _pool(self, last_hidden, attention_mask):
        cls = last_hidden[:, 0, :]
        if self.pooling == "cls":
            return cls
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-6)
        mean = summed / counts
        if self.pooling == "mean":
            return mean
        return torch.cat([cls, mean], dim=-1)          # cls_mean

    def encode(self, sentences: list[str], tokenizer, max_length: int) -> torch.Tensor:
        """Chunked encoding (your OOM-safe pattern), grad flows if fine-tuning."""
        embs = []
        for start in range(0, len(sentences), self.bert_batch_size):
            chunk = sentences[start: start + self.bert_batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(self.device_str)
            out = self.bert(**enc)
            embs.append(self._pool(out.last_hidden_state, enc["attention_mask"]))
        return torch.cat(embs, dim=0)
