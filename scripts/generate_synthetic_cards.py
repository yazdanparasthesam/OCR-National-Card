"""Generate a synthetic national-card dataset (images + ground truth JSON).

Examples::

    python scripts/generate_synthetic_cards.py --count 10 --difficulty clean --out data/samples
    python scripts/generate_synthetic_cards.py --count 40 --difficulty photo --out data/bench/photo
    python scripts/generate_synthetic_cards.py --count 40 --difficulty hard  --out data/bench/hard

The cards follow ``nidocr/templates/national_card_front.json``: the same geometry
the OCR pipeline uses, so the benchmark measures the pipeline and not a mismatch
between renderer and template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nidocr.config import PROJECT_ROOT  # noqa: E402
from nidocr.synthetic.renderer import DIFFICULTIES, save_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "samples"))
    parser.add_argument("--difficulty", default="photo", choices=DIFFICULTIES)
    parser.add_argument("--template", default="national_card_front")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="card")
    args = parser.parse_args()

    records = save_dataset(
        args.out,
        count=args.count,
        difficulty=args.difficulty,
        template=args.template,
        seed=args.seed,
        prefix=args.prefix,
    )
    print(f"wrote {len(records)} cards to {args.out} ({args.difficulty})")
    for record in records[:3]:
        print(f"  {record['image']}: {record['fields']['first_name']} {record['fields']['last_name']} "
              f"· {record['fields']['national_code']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
