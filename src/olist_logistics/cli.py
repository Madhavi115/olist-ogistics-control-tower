"""Command-line entry point for the Phase 1 pipeline."""

import argparse
import json
from pathlib import Path

from .pipeline import build_kpis, build_order_mart, load_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Olist Phase 1 analytical dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    orders, items, customers = load_sources(args.raw_dir)
    mart = build_order_mart(orders, items, customers)
    metrics = build_kpis(mart)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(args.out_dir / "order_mart.parquet", index=False)
    (args.report_dir / "phase1_kpis.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.out_dir / 'order_mart.parquet'}")


if __name__ == "__main__":
    main()
