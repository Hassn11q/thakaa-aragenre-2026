"""Build the ID pools consumed by scripts/run_full_taxonomy_verifiers.py.

  all_ids.json          every test id (input to the GPT full-context pass)
  gpt_disagree_ids.json ids where the GPT pass disagrees with the base judge
                        (input to the cheaper, targeted Gemini pass)
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--test", type=Path, default=ROOT / "test" / "test.json")
ap.add_argument("--base", type=Path, default=ROOT / "cached_outputs" / "base_predictions.json")
ap.add_argument("--gpt", type=Path, help="GPT full-context output; if given, also write gpt_disagree_ids.json")
ap.add_argument("--outdir", type=Path, default=ROOT / "work")
a = ap.parse_args()

a.outdir.mkdir(parents=True, exist_ok=True)
ids = [r["id"] for r in json.loads(a.test.read_text())]
(a.outdir / "all_ids.json").write_text(json.dumps(ids))
print(f"[pools] all_ids.json: {len(ids)}")

if a.gpt and a.gpt.exists():
    base = {x["id"]: x["specific_genre"] for x in json.loads(a.base.read_text())}
    gpt = json.loads(a.gpt.read_text())
    dis = [i for i in ids if gpt.get(i) and gpt[i] != base.get(i)]
    (a.outdir / "gpt_disagree_ids.json").write_text(json.dumps(dis))
    print(f"[pools] gpt_disagree_ids.json: {len(dis)}")
