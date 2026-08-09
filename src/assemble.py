"""Final assembly of the winning AraGenre 2026 submission (Thaka, 0.7352 Hierarchical Macro F1).

Takes the base judge predictions and applies, in order:
  1. Full-context two-model consensus corrections (GPT-5.6 Luna AND Gemini-3.6 Flash must agree
     on the same specific genre, and disagree with the base). Broad-changing moves are gated to
     read-verified-safe directions; broad-preserving refinements are always applied.
  2. Attractor drain: within-family redistribution of over-predicted generic classes using the
     chain-of-thought judge pass (broad-locked).
  3. High-precision surface rules (quran mushaf orthography, hadith isnad, Interactive markers).

Rebuilds submissions/final_submission.json exactly from the cached model outputs.

Usage:
  python src/assemble.py \
      --base artifacts/base_predictions.json \
      --gpt artifacts/verify_gpt.json \
      --gemini artifacts/verify_gemini.json \
      --cot artifacts/cot_predictions.json \
      --out work/predictions.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rules as AR  # noqa: E402

# Classes that absorb texts the judge cannot confidently place (found by manual reading).
ATTRACTORS = {
    "egyptian_song_lyrics",
    "literature_fiction_book_description",
    "history_encyclopedia",
    "culture_encyclopedia",
    "msa_poetry",
}

# Broad-changing consensus moves we verified by reading; all other cross-family moves are
# rejected because the models share a topic bias (e.g. Informative->Learning over-fires
# student-writing on academic prose).
SAFE_BROAD_MOVES = {
    ("Informative", "Interactive"),  # dialectal comments filed as book descriptions
    ("Legal", "Informative"),  # wire/agency news filed as resolutions or contracts
    ("Creative", "Informative"),  # long prose filed as song lyrics
    ("Informative", "Religious"),  # actual scripture filed as fiction/encyclopedia
    ("Creative", "Religious"),
    ("Interactive", "Religious"),
    ("Legal", "Interactive"),
}


def main():
    """Apply consensus corrections and the attractor drain to the base predictions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.json")
    ap.add_argument("--defs", type=Path, default=ROOT / "data" / "test_genre_definitions.json")
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="base judge predictions (artifacts/base_predictions.json)",
    )
    ap.add_argument("--gpt", type=Path, required=True, help="full-context GPT predictions")
    ap.add_argument("--gemini", type=Path, required=True, help="full-context Gemini predictions")
    ap.add_argument("--cot", type=Path, required=True, help="chain-of-thought judge predictions")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    defs = json.loads(a.defs.read_text())
    SPEC_TO_BROAD = {d["specific_genre"]: d["broad_genre"] for d in defs}
    family = defaultdict(set)
    for d in defs:
        family[d["broad_genre"]].add(d["specific_genre"])

    rows = json.loads(a.test.read_text())
    ids = [r["id"] for r in rows]
    text = {r["id"]: (r.get("text") or "") for r in rows}
    base = {x["id"]: dict(x) for x in json.loads(a.base.read_text())}
    cot = {x["id"]: dict(x) for x in json.loads(a.cot.read_text())}
    gpt = json.loads(a.gpt.read_text())
    gem = json.loads(a.gemini.read_text())

    out, counts = [], Counter()
    for i in ids:
        pred = dict(base[i])
        broad, spec = pred["broad_genre"], pred["specific_genre"]

        gpt_spec = gpt.get(i)
        consensus = gpt_spec in SPEC_TO_BROAD and gpt_spec == gem.get(i) and gpt_spec != spec

        if consensus:
            new_broad = SPEC_TO_BROAD[gpt_spec]
            if new_broad == broad:  # broad-preserving refinement
                pred = {"id": i, "broad_genre": new_broad, "specific_genre": gpt_spec}
                counts["spec_fix"] += 1
            elif (broad, new_broad) in SAFE_BROAD_MOVES:  # verified-safe broad move
                pred = {"id": i, "broad_genre": new_broad, "specific_genre": gpt_spec}
                counts["broad_fix"] += 1
            elif spec in ATTRACTORS:  # else fall back to drain
                sib = cot[i]["specific_genre"]
                if sib != spec and sib in family[broad]:
                    pred = {"id": i, "broad_genre": broad, "specific_genre": sib}
                    counts["drain"] += 1
        elif spec in ATTRACTORS:
            sib = cot[i]["specific_genre"]
            if sib != spec and sib in family[broad]:
                pred = {"id": i, "broad_genre": broad, "specific_genre": sib}
                counts["drain"] += 1

        rule = AR.rules(text[i])  # high-precision overrides
        if rule:
            pred = {
                "id": i,
                "broad_genre": SPEC_TO_BROAD[rule[1]],
                "specific_genre": rule[1],
            }

        out.append(
            {
                "id": i,
                "broad_genre": pred["broad_genre"],
                "specific_genre": pred["specific_genre"],
            }
        )

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    changed = sum(o["specific_genre"] != base[o["id"]]["specific_genre"] for o in out)
    print(f"[final] {dict(counts)} | specific changed vs base: {changed}/{len(ids)}")
    print(f"[final] broad: {Counter(o['broad_genre'] for o in out).most_common()}")
    print(f"[final] wrote {a.out}")


if __name__ == "__main__":
    main()
