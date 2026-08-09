"""Build the id pools consumed by the consensus verifiers (src/verify.py).

all_ids.json          every test id, the input to the full-context GPT pass
gpt_disagree_ids.json ids where that pass disagrees with the base judge, which is the
                      smaller pool the Gemini pass then re-checks
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.json")
    ap.add_argument(
        "--base", type=Path, default=ROOT / "artifacts" / "base_predictions.json"
    )
    ap.add_argument(
        "--gpt",
        type=Path,
        help="full-context GPT output; when given, also write gpt_disagree_ids.json",
    )
    ap.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    return ap.parse_args()


def write_all_ids(test_file, outdir):
    """Write every test id and return the list."""
    ids = [row["id"] for row in json.loads(test_file.read_text())]
    (outdir / "all_ids.json").write_text(json.dumps(ids))
    print(f"[pools] all_ids.json: {len(ids)}")
    return ids


def write_disagreements(ids, base_file, gpt_file, outdir):
    """Write the ids where the GPT pass and the base judge choose different genres."""
    base = {x["id"]: x["specific_genre"] for x in json.loads(base_file.read_text())}
    gpt = json.loads(gpt_file.read_text())
    disagreeing = [i for i in ids if gpt.get(i) and gpt[i] != base.get(i)]
    (outdir / "gpt_disagree_ids.json").write_text(json.dumps(disagreeing))
    print(f"[pools] gpt_disagree_ids.json: {len(disagreeing)}")


def main():
    """Write the verifier id pools."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    ids = write_all_ids(args.test, args.outdir)
    if args.gpt and args.gpt.exists():
        write_disagreements(ids, args.base, args.gpt, args.outdir)


if __name__ == "__main__":
    main()
