"""Check the judge-reproduction figures the paper reports for the base predictions.

artifacts/base_predictions.json is not a plain src/judge.py output: it also carries the
src/interactive_judge.py reassignment applied during the evaluation phase. This script applies
judge.py's own decision rule to the cached scores in artifacts/judge_rerun_scores.json and
reports how much of the base file a plain judge run recovers. The paper quotes 93.5% of the
specific labels and 97.1% of the broad labels.

Run:  python3 src/judge_rerun_check.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALPHA = 0.5  # the value the submitted run used; see src/judge.py


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, default=ROOT / "artifacts" / "judge_rerun_scores.json")
    ap.add_argument("--base", type=Path, default=ROOT / "artifacts" / "base_predictions.json")
    ap.add_argument("--defs", type=Path, default=ROOT / "data" / "test_genre_definitions.json")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()

    spec_to_broad = {
        d["specific_genre"]: d["broad_genre"] for d in json.loads(a.defs.read_text())
    }
    base = {x["id"]: x for x in json.loads(a.base.read_text())}
    per = {r["id"]: (r["genres"], r["scores"]) for r in json.loads(a.scores.read_text())}

    # judge.py subtracts a scaled per-genre mean score before taking the argmax
    total, count = {}, {}
    for genres, scores in per.values():
        for g, s in zip(genres, scores):
            total[g] = total.get(g, 0.0) + s
            count[g] = count.get(g, 0) + 1
    prior = {g: total[g] / count[g] for g in total}

    spec_hits = broad_hits = 0
    for rid, (genres, scores) in per.items():
        adjusted = [s - a.alpha * prior.get(g, 0.0) for g, s in zip(genres, scores)]
        pick = genres[max(range(len(genres)), key=adjusted.__getitem__)]
        spec_hits += pick == base[rid]["specific_genre"]
        broad_hits += spec_to_broad[pick] == base[rid]["broad_genre"]

    n = len(per)
    print(f"rows scored: {n}   alpha: {a.alpha}")
    print(f"specific labels recovered: {spec_hits}/{n} = {100 * spec_hits / n:.1f}%")
    print(f"broad labels recovered:    {broad_hits}/{n} = {100 * broad_hits / n:.1f}%")
    print("The shortfall is the Interactive-recall reassignment; see artifacts/README.md.")


if __name__ == "__main__":
    main()
