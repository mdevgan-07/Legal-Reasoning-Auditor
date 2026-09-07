"""
Central experiment configuration. Every experiment is an ExperimentConfig;
presets encode each tier so a full run is one CLI flag:

    python scripts/train_rrl.py --preset tierA

All experiments log to the same results.csv via evaluate.append_result, so
every variant is directly comparable (macro F1, per-class F1, val-test gap).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    # ---- identity -----------------------------------------------------
    name: str = "baseline"

    # ---- paths (defaults match your workspace) -------------------------
    data_dir: str = "/workspace/legal_capstone/data/Hier_BiLSTM_CRF"
    tag2idx_path: str = "/workspace/legal_capstone/saved_models/tag2idx.json"
    bert_dir: str = "/workspace/legal_capstone/saved_models/InLegalBERT_safe"
    out_dir: str = "/workspace/legal_capstone/rrl_experiments"
    init_ckpt: str = ""            # optional warm start (e.g. finetuned_bert_best.tar)

    # ---- data ----------------------------------------------------------
    # windowing: how to handle documents longer than max_sents
    #   truncate_head — first max_sents (YOUR OLD SETUP; loses document ends)
    #   head_tail     — first n_head + last n_tail (keeps Decisions at doc end)
    #   full          — keep everything up to hard_cap (chunked encoding makes
    #                   memory a non-issue; the 200 cap was a fine-tune-era relic)
    window: str = "head_tail"
    max_sents: int = 200
    n_head: int = 150
    n_tail: int = 50
    hard_cap: int = 350

    # ---- encoder ---------------------------------------------------------
    max_length: int = 256            # token cap per (context-augmented) sentence
    pooling: str = "cls_mean"        # cls | mean | cls_mean
    context: int = 1                 # neighbors each side encoded with [SEP] (0 = none)
    freeze_bert: bool = False        # True for two-stage stage-2 / cached-embedding runs
    n_fine_tune_layers: int = 12
    bert_batch_size: int = 16
    grad_checkpoint: bool = False    # BERT gradient checkpointing (needed for window=full)
    lora: bool = False               # Tier C: LoRA adapters instead of full fine-tune
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- sequence layer ---------------------------------------------------
    sequence: str = "bilstm"         # bilstm | transformer
    hidden_dim: int = 512
    lstm_dropout: float = 0.5
    tf_layers: int = 3               # transformer sequence-encoder depth
    tf_heads: int = 8
    tf_ff: int = 1024
    tf_dropout: float = 0.2
    positional: bool = True          # sinusoidal positions for transformer

    # ---- loss --------------------------------------------------------------
    class_weights: tuple = (0.0, 0.0, 0.0, 0.4, 1.0, 5.0, 2.0, 2.5, 1.2, 4.0)
    # weight_scheme: manual (use class_weights above) | effective | inverse_sqrt
    weight_scheme: str = "manual"
    cb_beta: float = 0.9999          # effective-number beta
    focal_lambda: float = 0.5        # weight of auxiliary focal CE on emissions (0 = off)
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0     # smoothing inside the focal aux loss
    # aux None-vs-content head: joint binary task targeting the Reasoning<->None
    # boundary (the dominant confusion). 0 = off.
    aux_none_lambda: float = 0.0
    # document-level oversampling of rare-class-rich docs (1.0 = off)
    oversample_boost: float = 1.0
    rare_classes: tuple = ("Issue", "Decision")

    # ---- optimization ---------------------------------------------------------
    epochs: int = 25
    lr_bert: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    grad_accum_docs: int = 32
    warmup_frac: float = 0.1
    patience: int = 5                # early stop on val macro F1
    seed: int = 42

    # ---- checkpointing ------------------------------------------------------
    keep_top_k: int = 3              # keep k best-val checkpoints for averaging
    average_top_k: bool = True       # evaluate the averaged model too

    # ---- two-stage / cached embeddings ------------------------------------
    emb_cache_dir: str = ""          # set => stage-2 mode: train sequence+CRF on cache
    stage2_batch_docs: int = 16
    stage2_epochs: int = 60
    stage2_lr: float = 1e-3

    # ---- post-hoc ------------------------------------------------------------
    apply_constraints: bool = False  # structural constraint pass at inference

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets — each tier is a named delta on the baseline.
# ---------------------------------------------------------------------------

def _mk(name: str, **kw) -> ExperimentConfig:
    c = ExperimentConfig(name=name)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


PRESETS: dict[str, ExperimentConfig] = {
    # Faithful reproduction of your current setup — the control.
    "baseline": _mk(
        "baseline", window="truncate_head", max_sents=200, max_length=128,
        pooling="cls", context=0, focal_lambda=0.0, keep_top_k=1,
        average_top_k=False, sequence="bilstm",
    ),

    # Diagnosis fix alone: windowing + longer tokens, nothing else.
    # Isolates how much the truncation bug was costing you.
    "fix_window": _mk(
        "fix_window", window="head_tail", max_length=256,
        pooling="cls", context=0, focal_lambda=0.0, keep_top_k=1,
        average_top_k=False,
    ),
    
    # Truncation fix done properly: (almost) nothing deleted. full@2000 covers
    # >99% of docs end-to-end; the 33 train docs over the cap keep head+tail
    # within it so Decision tails survive. Single-variable vs fix_window/baseline.
    "fix_full": _mk(
        "fix_full", window="full", hard_cap=2000, max_length=256,
        pooling="cls", context=0, focal_lambda=0.0, keep_top_k=1,
        average_top_k=False, grad_checkpoint=True,
    ),

    # Tier A: everything cheap at once.
    "tierA": _mk(
        "tierA", window="head_tail", max_length=256, pooling="cls_mean",
        context=1, focal_lambda=0.5, keep_top_k=3, average_top_k=True,
    ),

    # Tier A + imbalance toolkit: effective-number weights, doc oversampling,
    # label smoothing, and the aux None-vs-content head (targets the dominant
    # Reasoning<->None confusion from the diagnostic).
    "tierA_balance": _mk(
        "tierA_balance", window="head_tail", max_length=256, pooling="cls_mean",
        context=1, focal_lambda=0.5, keep_top_k=3, average_top_k=True,
        weight_scheme="effective", oversample_boost=2.0,
        label_smoothing=0.05, aux_none_lambda=0.3,
    ),

    # Tier B1: hierarchical Transformer sequence layer (keep Tier A gains).
    "tierB_transformer": _mk(
        "tierB_transformer", window="head_tail", max_length=256,
        pooling="cls_mean", context=1, focal_lambda=0.5,
        sequence="transformer", tf_layers=3, keep_top_k=3, average_top_k=True,
    ),

    # Tier B2 stage-2: frozen cached embeddings, big batches, long training.
    # (Run scripts/precompute_embeddings.py first; pass --emb-cache.)
    "tierB_twostage": _mk(
        "tierB_twostage", freeze_bert=True, sequence="bilstm",
        focal_lambda=0.5, keep_top_k=3, average_top_k=True,
    ),

    # Tier C: LoRA adapters on InLegalBERT instead of full fine-tuning.
    "tierC_lora": _mk(
        "tierC_lora", window="head_tail", max_length=256, pooling="cls_mean",
        context=1, focal_lambda=0.5, lora=True, n_fine_tune_layers=0,
        keep_top_k=3, average_top_k=True,
    ),
}
