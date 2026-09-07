"""Assemble the Thakaa submission that scored 0.7352 Hierarchical Macro F1.

The script applies three steps to the base predictions:
  1. Full-context consensus corrections. GPT-5.6 Luna and Gemini-3.6 Flash must agree on the same
     specific genre and disagree with the base. Broad-changing moves are limited to directions
     checked by manual reading; broad-preserving changes are accepted.
  2. Attractor drain: within-family redistribution of over-predicted generic classes using the
     chain-of-thought judge pass (broad-locked).
  3. Surface-rule overrides for Quranic orthography, hadith isnad, and Interactive markers. The
     offline reconstruction reads these from a text-free cache because the official test text
     cannot be redistributed.

Reproduces submissions/final_submission.json exactly from the cached model outputs.

Usage:
  python scripts/build_final.py \
      --base cached_outputs/base_predictions.json \
      --gpt cached_outputs/gpt_verifier_predictions.json \
      --gemini cached_outputs/gemini_verifier_predictions.json \
      --cot cached_outputs/rationalized_judge_predictions.json \
      --rule-overrides cached_outputs/surface_rule_overrides.json \
      --out build/predictions.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Classes that absorb texts the judge cannot confidently place (found by manual reading).
ATTRACTORS = {
    "egyptian_song_lyrics",
    "literature_fiction_book_description",
    "history_encyclopedia",
    "culture_encyclopedia",
    "msa_poetry",
}

# Broad-changing consensus moves checked by reading; all other cross-family moves are
# rejected because the models share a topic bias (e.g. Informative->Learning over-fires
# student-writing on academic prose).
SAFE_BROAD_MOVES = {
    ("Informative", "Interactive"),   # dialectal comments filed as book descriptions
    ("Legal", "Informative"),         # wire/agency news filed as resolutions or contracts
    ("Creative", "Informative"),      # long prose filed as song lyrics
    ("Informative", "Religious"),     # actual scripture filed as fiction/encyclopedia
    ("Creative", "Religious"),
    ("Interactive", "Religious"),
    ("Legal", "Interactive"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", type=Path, default=ROOT / "test" / "test_genre_definitions.json")
    ap.add_argument("--base", type=Path, required=True, help="base judge predictions")
    ap.add_argument("--gpt", type=Path, required=True, help="full-context GPT predictions")
    ap.add_argument("--gemini", type=Path, required=True, help="full-context Gemini predictions")
    ap.add_argument("--cot", type=Path, required=True, help="chain-of-thought judge predictions")
    ap.add_argument(
        "--rule-overrides",
        type=Path,
        required=True,
        help="text-free cache of the surface-rule outputs used in the submitted run",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    defs = json.loads(a.defs.read_text())
    s2b = {d["specific_genre"]: d["broad_genre"] for d in defs}
    family = defaultdict(set)
    for d in defs:
        family[d["broad_genre"]].add(d["specific_genre"])

    base_rows = json.loads(a.base.read_text())
    ids = [r["id"] for r in base_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("base predictions contain duplicate IDs")
    base = {x["id"]: dict(x) for x in base_rows}
    cot = {x["id"]: dict(x) for x in json.loads(a.cot.read_text())}
    gpt = json.loads(a.gpt.read_text())
    gem = json.loads(a.gemini.read_text())
    rule_overrides = {x["id"]: dict(x) for x in json.loads(a.rule_overrides.read_text())}
    unknown_override_ids = set(rule_overrides) - set(ids)
    if unknown_override_ids:
        raise ValueError(f"surface-rule cache contains {len(unknown_override_ids)} unknown IDs")

    out, counts = [], Counter()
    for i in ids:
        pred = dict(base[i])
        broad, spec = pred["broad_genre"], pred["specific_genre"]

        gpt_spec = gpt.get(i)
        consensus = gpt_spec in s2b and gpt_spec == gem.get(i) and gpt_spec != spec

        if consensus:
            new_broad = s2b[gpt_spec]
            if new_broad == broad:                                   # broad-preserving refinement
                pred = {"id": i, "broad_genre": new_broad, "specific_genre": gpt_spec}
                counts["spec_fix"] += 1
            elif (broad, new_broad) in SAFE_BROAD_MOVES:             # verified-safe broad move
                pred = {"id": i, "broad_genre": new_broad, "specific_genre": gpt_spec}
                counts["broad_fix"] += 1
            elif spec in ATTRACTORS:                                 # else fall back to drain
                sib = cot[i]["specific_genre"]
                if sib != spec and sib in family[broad]:
                    pred = {"id": i, "broad_genre": broad, "specific_genre": sib}
                    counts["drain"] += 1
        elif spec in ATTRACTORS:
            sib = cot[i]["specific_genre"]
            if sib != spec and sib in family[broad]:
                pred = {"id": i, "broad_genre": broad, "specific_genre": sib}
                counts["drain"] += 1

        if i in rule_overrides:                                      # submitted rule outputs
            spec = rule_overrides[i]["specific_genre"]
            if spec not in s2b:
                raise ValueError(f"surface-rule cache has an unknown label for ID {i!r}: {spec!r}")
            pred = {"id": i, "broad_genre": s2b[spec], "specific_genre": spec}

        out.append({"id": i, "broad_genre": pred["broad_genre"], "specific_genre": pred["specific_genre"]})

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    changed = sum(o["specific_genre"] != base[o["id"]]["specific_genre"] for o in out)
    print(f"[final] {dict(counts)} | specific changed vs base: {changed}/{len(ids)}")
    print(f"[final] broad: {Counter(o['broad_genre'] for o in out).most_common()}")
    print(f"[final] wrote {a.out}")


if __name__ == "__main__":
    main()
