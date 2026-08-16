"""Rebuild the three-arm topic-trap table from cached verifier outputs.

The full-taxonomy prompt carries an explicit type-versus-topic instruction that the
broad-only prompt lacks, so comparing those two arms alone confounds "more label context"
with "an explicit anti-trap rule". The third arm separates them:

  A  broad definitions only, no instruction      artifacts/trap_arm_a.json
  B  broad definitions only, plus the instruction artifacts/trap_arm_b.json
  C  all 74 definitions, plus the instruction     artifacts/trap_arm_c.json

A -> B isolates the instruction, B -> C isolates the taxonomy. All three arms use the same model and
the same 2,635-instance pool (artifacts/trap_pool.json), so the comparison is like for like;
artifacts/verify_gpt.json is the separate all-instance run used by the deployed pipeline; the table counts how often each arm files
a book description under Religious.

Run:  python3 src/trap_table.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=Path, default=ROOT / "artifacts" / "trap_pool.json")
    ap.add_argument("--arm-a", type=Path, default=ROOT / "artifacts" / "trap_arm_a.json")
    ap.add_argument("--arm-b", type=Path, default=ROOT / "artifacts" / "trap_arm_b.json")
    ap.add_argument("--arm-c", type=Path, default=ROOT / "artifacts" / "trap_arm_c.json")
    ap.add_argument("--base", type=Path, default=ROOT / "artifacts" / "base_predictions.json")
    ap.add_argument("--defs", type=Path, default=ROOT / "data" / "test_genre_definitions.json")
    a = ap.parse_args()

    spec_to_broad = {
        d["specific_genre"]: d["broad_genre"] for d in json.loads(a.defs.read_text())
    }
    base = {x["id"]: x for x in json.loads(a.base.read_text())}
    pool = json.loads(a.pool.read_text())

    # arms A and B answer with a broad genre; arm C answers with a specific genre
    arms = [
        ("A  broad defs only", json.loads(a.arm_a.read_text()), False),
        ("B  + type-vs-topic instruction", json.loads(a.arm_b.read_text()), False),
        ("C  + full 74-definition taxonomy", json.loads(a.arm_c.read_text()), True),
    ]

    book = [i for i in pool if base[i]["specific_genre"].endswith("_book_description")]
    print(f"pool {len(pool)}, of which book descriptions {len(book)}\n")
    print(f"{'arm':36s}{'called Religious':>18s}")
    for name, verdicts, is_specific in arms:
        religious = 0
        for i in book:
            v = verdicts.get(i)
            if v is None:
                continue
            broad = spec_to_broad.get(v, v) if is_specific else v
            religious += broad == "Religious"
        print(f"{name:36s}{religious:18d}")


if __name__ == "__main__":
    main()
