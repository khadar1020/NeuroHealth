#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurohealth.phase1.pipeline import Step1Config, run_phase1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuroHealth Phase 1 data pipeline.")
    parser.add_argument("--project-root", default=str(ROOT), help="Project root path")
    parser.add_argument("--max-medlineplus-codes", type=int, default=100)
    parser.add_argument("--stackexchange-pages", type=int, default=10)
    parser.add_argument("--stackexchange-page-size", type=int, default=100)
    parser.add_argument("--synthetic-dialogue-count", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Step1Config(
        project_root=args.project_root,
        max_medlineplus_codes=args.max_medlineplus_codes,
        stackexchange_pages=args.stackexchange_pages,
        stackexchange_page_size=args.stackexchange_page_size,
        synthetic_dialogue_count=args.synthetic_dialogue_count,
    )
    result = run_phase1_pipeline(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
