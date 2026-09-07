#!/usr/bin/env python3
"""
Unified experiment runner. Each preset is one overnight run; all results land
in <out_dir>/results.csv for direct comparison.

  python scripts/train_rrl.py --preset baseline            # control (your old setup)
  python scripts/train_rrl.py --preset fix_window          # truncation fix ONLY
  python scripts/train_rrl.py --preset tierA               # all cheap wins
  python scripts/train_rrl.py --preset tierB_transformer   # transformer sequence layer
  python scripts/train_rrl.py --preset tierC_lora          # LoRA encoder (pip install peft)

Any config field is overridable:
  python scripts/train_rrl.py --preset tierA --set epochs=30 seed=7 focal_gamma=1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrl.config import PRESETS
from rrl.train import Trainer


def parse_overrides(pairs: list[str], cfg) -> None:
    for pair in pairs:
        k, v = pair.split("=", 1)
        if not hasattr(cfg, k):
            sys.exit(f"unknown config field: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            setattr(cfg, k, v.lower() in ("1", "true", "yes"))
        elif isinstance(cur, int):
            setattr(cfg, k, int(v))
        elif isinstance(cur, float):
            setattr(cfg, k, float(v))
        else:
            setattr(cfg, k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--name", default=None, help="override run name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", nargs="*", default=[], help="field=value overrides")
    args = ap.parse_args()

    import copy
    cfg = copy.deepcopy(PRESETS[args.preset])
    if args.name:
        cfg.name = args.name
    parse_overrides(args.set, cfg)

    print(f"=== RUN {cfg.name} ===")
    Trainer(cfg, device=args.device).train()


if __name__ == "__main__":
    main()
