#!/usr/bin/env python3
"""Rebuild the prompt-provenance diagnostics reported in the paper.

The cached files contain only instance IDs and predicted labels. No hidden-test text or gold
labels are required. Row groups are defined by the submitted base predictions, so these are
agreement diagnostics rather than accuracy estimates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "cached_outputs"
DIAGNOSTICS = CACHE_ROOT / "diagnostics"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pool = load(DIAGNOSTICS / "verification_pool_ids.json")
    base = {row["id"]: row for row in load(CACHE_ROOT / "base_predictions.json")}
    definitions = load(ROOT / "test" / "test_genre_definitions.json")
    specific_to_broad = {row["specific_genre"]: row["broad_genre"] for row in definitions}

    broad_only = load(DIAGNOSTICS / "gpt_broad_only_predictions.json")
    instruction = load(DIAGNOSTICS / "gpt_type_instruction_predictions.json")
    full_taxonomy = load(DIAGNOSTICS / "gpt_pool_full_taxonomy_predictions.json")

    books = [i for i in pool if base[i]["specific_genre"].endswith("book_description")]
    religious = [i for i in pool if base[i]["broad_genre"] == "Religious"]
    excluded = set(books) | set(religious)
    rest = [i for i in pool if i not in excluded]

    arms = {
        "A_broad_only": broad_only,
        "B_type_instruction": instruction,
        "C_full_taxonomy": {i: specific_to_broad.get(full_taxonomy.get(i)) for i in pool},
    }
    print("Three-arm Religious calls")
    print("arm\tbook_descriptions\trest_of_pool\tbase_religious\tseparation_pp")
    for name, predictions in arms.items():
        book_calls = sum(predictions.get(i) == "Religious" for i in books)
        rest_calls = sum(predictions.get(i) == "Religious" for i in rest)
        religious_calls = sum(predictions.get(i) == "Religious" for i in religious)
        other_rate = (book_calls + rest_calls) / (len(books) + len(rest))
        religious_rate = religious_calls / len(religious)
        separation = round(100 * (religious_rate - other_rate))
        print(f"{name}\t{book_calls}\t{rest_calls}\t{religious_calls}\t{separation}")

    by_family: dict[str, list[str]] = defaultdict(list)
    for i in pool:
        by_family[base[i]["broad_genre"]].append(i)

    print("\nBase-family agreement")
    print("family\tn\tbroad_only\tfull_taxonomy")
    for family in ("Informative", "Learning", "Interactive", "Legal", "Creative", "Religious"):
        ids = by_family[family]
        a = sum(broad_only.get(i) == family for i in ids) / len(ids)
        c = sum(specific_to_broad.get(full_taxonomy.get(i)) == family for i in ids) / len(ids)
        print(f"{family}\t{len(ids)}\t{a:.0%}\t{c:.0%}")

    gemini_broad = load(DIAGNOSTICS / "gemini_broad_only_predictions.json")
    gemini_full = load(CACHE_ROOT / "gemini_verifier_predictions.json")
    gpt_deployed = load(CACHE_ROOT / "gpt_verifier_predictions.json")
    paired = [i for i in books if gemini_broad.get(i) is not None and gemini_full.get(i) is not None]
    broad_joint = sum(
        broad_only.get(i) == "Religious" and gemini_broad.get(i) == "Religious" for i in paired
    )
    full_joint = sum(
        specific_to_broad.get(gpt_deployed.get(i)) == "Religious"
        and specific_to_broad.get(gemini_full.get(i)) == "Religious"
        for i in paired
    )
    print("\nTwo-verifier paired comparison")
    print(f"paired_book_descriptions\t{len(paired)}")
    print(f"joint_religious_broad_only\t{broad_joint}")
    print(f"joint_religious_full_taxonomy\t{full_joint}")


if __name__ == "__main__":
    main()
