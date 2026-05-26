"""Minimal smoke evaluator without optional plotting dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    result_file = Path(args.save_dir) / "results.csv"
    df = pd.read_csv(result_file)
    acc = float(df["qa_acc"].mean()) if "qa_acc" in df and len(df) else 0.0
    summary_file = Path(args.save_dir) / "summary.txt"
    summary_file.write_text(f"samples={len(df)}\nqa_acc={acc:.2f}\n", encoding="utf-8")
    print(f"SMOKE_EVAL samples={len(df)} qa_acc={acc:.2f}")


if __name__ == "__main__":
    main()
