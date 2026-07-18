#!/usr/bin/env python
"""Placeholder for late_enrollment experiment.
Accepts a seed argument and creates a dummy log file.
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / f"late_enrollment_seed{args.seed}.log"
    log_file.write_text(f"Placeholder late_enrollment executed with seed {args.seed}\n")
    print(f"[OK] late_enrollment seed={args.seed} placeholder executed")

if __name__ == "__main__":
    main()
