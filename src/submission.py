#!/usr/bin/env python3
"""Build a competition-ready AraGenre submission ZIP from a predictions file.

Hidden-test-safe packaging. Given the evaluation input file, the split's genre
definitions, and a raw predictions file, this script:

  1. Re-orders predictions to the exact input ID order (Codabench keys by id, but we
     preserve order defensively).
  2. Enforces hierarchy: broad_genre is OVERWRITTEN with the parent implied by
     specific_genre in the definitions file, so a submission can never be
     hierarchy-inconsistent.
  3. Validates every id is present exactly once, no extra ids, labels are exact names
     from the definitions file.
  4. Writes an ordered submission JSON and zips it (the ZIP the task requires).

It never edits label content beyond the deterministic broad<-specific derivation, and it
refuses to write a ZIP if any check fails. No dev labels or heuristics are involved.

Usage:
  python src/submission.py \
    --input-file  path/to/test.json \
    --definitions-file path/to/test_genre_definitions.json \
    --predictions-file outputs/experiments/<run>/work/predictions.json \
    --out-json submissions/test_final.json \
    --out-zip  submissions/test_final.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

# The six broad genres documented on the official task page. Used only for a warning; the
# authoritative mapping always comes from the released definitions file.
KNOWN_BROAD_GENRES = {
    "Informative",
    "Creative",
    "Interactive",
    "Learning",
    "Legal",
    "Religious",
}


def load_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_specific_to_broad(definitions: list[dict[str, Any]]) -> dict[str, str]:
    """Map every specific genre to its parent broad genre."""
    mapping: dict[str, str] = {}
    for i, row in enumerate(definitions):
        spec = row.get("specific_genre")
        broad = row.get("broad_genre")
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(f"definitions[{i}] has an invalid specific_genre.")
        if not isinstance(broad, str) or not broad.strip():
            raise ValueError(f"definitions[{i}] has an invalid broad_genre.")
        if spec in mapping and mapping[spec] != broad:
            raise ValueError(
                f"specific_genre {spec!r} maps to two broad genres: {mapping[spec]!r} and {broad!r}."
            )
        mapping[spec] = broad
    return mapping


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble validated prediction rows in the order of the input file."""
    inputs = load_json(args.input_file)
    definitions = load_json(args.definitions_file)
    predictions = load_json(args.predictions_file)

    input_ids = [row["id"] for row in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("Input file contains duplicate ids.")

    SPEC_TO_BROAD = build_specific_to_broad(definitions)
    allowed_specific = set(SPEC_TO_BROAD)

    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        rid = row.get("id")
        if rid is None:
            raise ValueError("A prediction row is missing 'id'.")
        if rid in pred_by_id:
            raise ValueError(f"Duplicate prediction id: {rid!r}.")
        pred_by_id[rid] = row

    missing = [rid for rid in input_ids if rid not in pred_by_id]
    if missing:
        raise ValueError(f"Predictions missing {len(missing)} ids, first: {missing[:10]}")

    ordered: list[dict[str, str]] = []
    fixed_broad = 0
    for rid in input_ids:  # exact input order, exactly the evaluation ids
        spec = pred_by_id[rid].get("specific_genre")
        if spec not in allowed_specific:
            raise ValueError(
                f"id {rid!r}: specific_genre {spec!r} is not one of the definition labels."
            )
        derived_broad = SPEC_TO_BROAD[spec]
        if pred_by_id[rid].get("broad_genre") != derived_broad:
            fixed_broad += 1
        ordered.append({"id": rid, "broad_genre": derived_broad, "specific_genre": spec})

    # The official task page fixes the broad taxonomy at six categories. Broad labels here
    # are derived from the released definitions file, so an out-of-set value means the
    # released taxonomy itself changed, not that the system misbehaved. Warn rather than
    # refuse: the organisers may legitimately extend the set, and failing the build would
    # block a valid submission during the five-day evaluation window.
    unexpected = sorted({row["broad_genre"] for row in ordered} - KNOWN_BROAD_GENRES)
    if unexpected:
        print(
            f"WARNING: broad genres outside the documented six: {unexpected}. "
            f"Confirm against the released taxonomy before submitting.",
            file=sys.stderr,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Store under a fixed arcname so the archive layout is stable/predictable.
        zf.write(args.out_json, arcname=args.arcname)

    return {
        "num_predictions": len(ordered),
        "num_input_ids": len(input_ids),
        "broad_genres_corrected_from_specific": fixed_broad,
        "out_json": str(args.out_json),
        "out_zip": str(args.out_zip),
        "arcname_in_zip": args.arcname,
    }


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Evaluation input JSON (ids + text).",
    )
    p.add_argument(
        "--definitions-file",
        type=Path,
        required=True,
        help="Split genre definitions JSON.",
    )
    p.add_argument("--predictions-file", type=Path, required=True, help="Raw predictions JSON.")
    p.add_argument("--out-json", type=Path, required=True, help="Ordered submission JSON to write.")
    p.add_argument("--out-zip", type=Path, required=True, help="Submission ZIP to write.")
    p.add_argument("--arcname", default="predictions.json", help="Filename inside the ZIP.")
    return p.parse_args()


def main() -> None:
    """Build a validated submission file and its zip archive."""
    args = parse_args()
    try:
        report = build(args)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OK: submission JSON + ZIP written and hierarchy-validated.")


if __name__ == "__main__":
    main()
