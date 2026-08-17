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
    religious = [i for i in pool if base[i]["broad_genre"] == "Religious"]
    rest = [i for i in pool if i not in set(book) | set(religious)]
    print(f"pool {len(pool)}: book {len(book)}, rest {len(rest)}, base-Religious {len(religious)}\n")

    def called_religious(ids, verdicts, is_specific):
        """How many of `ids` this arm puts under the Religious broad genre."""
        n = 0
        for i in ids:
            v = verdicts.get(i)
            if v is None:  # the model returned nothing parsable
                continue
            n += (spec_to_broad.get(v, v) if is_specific else v) == "Religious"
        return n

    print(f"{'Religious calls':34s}{'A':>8s}{'B':>8s}{'C':>8s}")
    for name, ids in [
        ("book descriptions", book),
        ("rest of pool", rest),
        ("base-Religious", religious),
    ]:
        counts = [called_religious(ids, v, s) for _, v, s in arms]
        print(f"{name + f' (n={len(ids)})':34s}" + "".join(f"{c:8d}" for c in counts))

    # separation: the Religious-call rate on base-Religious minus the rate on everything else.
    # A drop driven by a blanket bias away from Religious would leave this flat; it rises.
    other = book + rest
    seps = []
    for _, v, s in arms:
        hit = called_religious(religious, v, s) / len(religious)
        false = called_religious(other, v, s) / len(other)
        seps.append(100 * (hit - false))
    print(f"\n{'separation (pp)':34s}" + "".join(f"{x:8.0f}" for x in seps))
    print("\nBase labels are the system's own calls, not gold: read as agreement, not accuracy.")


if __name__ == "__main__":
    main()
