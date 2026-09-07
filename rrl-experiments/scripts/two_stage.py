#!/usr/bin/env python3
"""
Tier B two-stage, both stages.

Stage 1 — regenerate embeddings FRESH from a chosen encoder checkpoint
(clean experiment; does not reuse old .pkl caches):

  python scripts/two_stage.py precompute \
      --cache /workspace/legal_capstone/emb_cache_ft \
      --encoder-ckpt /workspace/legal_capstone/rrl_experiments/tierA/ckpt_eXX_fY.pt

  # or from base InLegalBERT (no fine-tuning) for the frozen-baseline arm:
  python scripts/two_stage.py precompute --cache .../emb_cache_base

Stage 2 — train sequence+CRF fast on the cache (minutes/epoch, real batches):

  python scripts/two_stage.py train --cache .../emb_cache_ft --name twostage_ft
  python scripts/two_stage.py train --cache .../emb_cache_ft --name twostage_ft_tf --set sequence=transformer
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrl.config import PRESETS
from rrl.embed_cache import precompute, train_stage2


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("precompute")
    p1.add_argument("--cache", required=True)
    p1.add_argument("--encoder-ckpt", default="")
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--set", nargs="*", default=[])

    p2 = sub.add_parser("train")
    p2.add_argument("--cache", required=True)
    p2.add_argument("--name", default="tierB_twostage")
    p2.add_argument("--device", default="cuda")
    p2.add_argument("--set", nargs="*", default=[])

    args = ap.parse_args()
    cfg = copy.deepcopy(PRESETS["tierB_twostage"])

    from train_rrl import parse_overrides  # same override parser
    parse_overrides(args.set, cfg)

    if args.cmd == "precompute":
        precompute(cfg, cache_dir=args.cache,
                   encoder_ckpt=args.encoder_ckpt, device=args.device)
    else:
        cfg.emb_cache_dir = args.cache
        cfg.name = args.name
        train_stage2(cfg, device=args.device)


if __name__ == "__main__":
    main()
