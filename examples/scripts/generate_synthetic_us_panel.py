"""Generate a deterministic synthetic US-like market panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fieldtrial.data.synthetic import generate_synthetic_us_panel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic_panel.parquet")
    parser.add_argument("--markets", type=int, default=96)
    parser.add_argument("--periods", type=int, default=730)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_us_panel(
        n_markets=args.markets,
        start=args.start,
        periods=args.periods,
        seed=args.seed,
    )
    frame.to_parquet(args.out, index=False)
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(args.out.resolve()),
                "rows": len(frame),
                "markets": args.markets,
                "periods": args.periods,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
