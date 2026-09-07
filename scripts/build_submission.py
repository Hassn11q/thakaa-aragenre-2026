#!/usr/bin/env python3
"""Build an AraGenre submission ZIP from a predictions file.

Given the evaluation input, genre definitions, and raw predictions, this script:

  1. Reorders predictions to the input ID order.
  2. Derives ``broad_genre`` from ``specific_genre`` using the definitions file.
  3. Checks that every ID appears once and that all labels occur in the definitions.
  4. Writes the ordered JSON and the required ZIP.

No development labels or heuristics are used during packaging.

Usage:
  python scripts/build_submission.py \
    --input-file  path/to/test.json \
    --definitions-file path/to/test_genre_definitions.json \
    --predictions-file outputs/experiments/<run>/predictions_dev.json \
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
    "Informative", "Creative", "Interactive", "Learning", "Legal", "Religious",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_specific_to_broad(definitions: list[dict[str, Any]]) -> dict[str, str]:
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
    definitions = load_json(args.definitions_file)
    predictions = load_json(args.predictions_file)

    if args.input_file is None:
        input_ids = [row["id"] for row in predictions]
    else:
        inputs = load_json(args.input_file)
        input_ids = [row["id"] for row in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("Input file contains duplicate ids.")

    s2b = build_specific_to_broad(definitions)
    allowed_specific = set(s2b)

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
        derived_broad = s2b[spec]
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-file",
        type=Path,
        help="Optional evaluation input JSON used to impose ID order. If omitted, prediction order is retained.",
    )
    p.add_argument("--definitions-file", type=Path, required=True, help="Split genre definitions JSON.")
    p.add_argument("--predictions-file", type=Path, required=True, help="Raw predictions JSON.")
    p.add_argument("--out-json", type=Path, required=True, help="Ordered submission JSON to write.")
    p.add_argument("--out-zip", type=Path, required=True, help="Submission ZIP to write.")
    p.add_argument("--arcname", default="predictions.json", help="Filename inside the ZIP.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = build(args)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OK: submission JSON + ZIP written and hierarchy-validated.")


if __name__ == "__main__":
    main()
