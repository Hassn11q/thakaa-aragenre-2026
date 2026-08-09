#!/bin/sh
# Rebuild the submitted predictions from cached model outputs and check them.
set -e
cd "$(dirname "$0")"

[ -f data/test.json ] || { echo "missing data/test.json (not redistributed; get it from the organisers)"; exit 1; }

python3 src/assemble.py \
  --base artifacts/base_predictions.json \
  --gpt artifacts/verify_gpt.json \
  --gemini artifacts/verify_gemini.json \
  --cot artifacts/cot_predictions.json \
  --out work/predictions.json

python3 src/submission.py \
  --input-file data/test.json \
  --definitions-file data/test_genre_definitions.json \
  --predictions-file work/predictions.json \
  --out-json work/final_submission.json \
  --out-zip work/final_submission.zip \
  --arcname predictions.json

python3 - <<'PY'
import json
a = json.load(open("work/final_submission.json"))
b = json.load(open("submissions/final_submission.json"))
same = sum(x == y for x, y in zip(a, b))
print(f"identical predictions: {same}/{len(b)}")
raise SystemExit(0 if same == len(b) else 1)
PY
