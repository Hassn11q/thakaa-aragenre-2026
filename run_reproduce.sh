#!/usr/bin/env bash
# Rebuild the winning AraGenre 2026 submission from cached model outputs and verify it.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

echo "[1/3] final assembly (consensus + drain + rules)"
python3 scripts/build_final.py \
  --defs test/test_genre_definitions.json \
  --base cached_outputs/base_predictions.json \
  --gpt cached_outputs/gpt_verifier_predictions.json \
  --gemini cached_outputs/gemini_verifier_predictions.json \
  --cot cached_outputs/rationalized_judge_predictions.json \
  --rule-overrides cached_outputs/surface_rule_overrides.json \
  --out  build/predictions.json

echo "[2/3] validate hierarchy + build zip"
python3 scripts/build_submission.py \
  --definitions-file test/test_genre_definitions.json \
  --predictions-file build/predictions.json \
  --out-json build/final_submission.json \
  --out-zip  build/final_submission.zip \
  --arcname  predictions.json

echo "[3/3] verify against the official 0.7352 submission"
python3 - <<'PY'
import json
a = {x["id"]: (x["broad_genre"], x["specific_genre"]) for x in json.load(open("build/final_submission.json"))}
b = {x["id"]: (x["broad_genre"], x["specific_genre"]) for x in json.load(open("submissions/final_submission.json"))}
same = sum(a[k] == b[k] for k in b)
print(f"identical predictions: {same}/{len(b)}")
raise SystemExit(0 if same == len(b) and len(a) == len(b) else 1)
PY
echo "OK: reproduction matches the submitted system."
