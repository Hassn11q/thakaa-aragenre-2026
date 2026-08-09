"""Disentangle the two factors in the topic-trap experiment.

Appendix D compared (A) broad-only definitions vs (C) full 74-definition taxonomy, but the
full-taxonomy prompt also carried an explicit type-vs-topic instruction that the broad-only
prompt lacked. The comparison therefore confounds "more label context" with "an explicit
anti-trap rule". This script runs the missing arm:

  B = broad-only definitions + the SAME explicit anti-trap instruction

Comparing A -> B isolates the instruction, and B -> C isolates the taxonomy.

  PROVIDER=gpt python src/ablate_trap.py
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"
POOL = Path(os.environ.get("POOL", ROOT / "work" / "broad_pool.json"))
OUT = Path(os.environ.get("OUT", ROOT / "work" / "trap_arm_b.json"))
WORKERS = int(os.environ.get("WORKERS", "24"))

defs = json.loads(DEFS.read_text())
BROADS, seen = [], set()
for d in defs:
    if d["broad_genre"] not in seen:
        seen.add(d["broad_genre"])
        BROADS.append((d["broad_genre"], d["broad_genre_definition"]))
BLOCK = "\n".join(f"- {b}: {t}" for b, t in BROADS)
NAMES = [b for b, _ in BROADS]

# Arm B: broad-only definitions (same as arm A) PLUS the anti-trap instruction that arm C had.
SYS = (
    "You are an expert Arabic text-genre analyst. Classify the text into exactly ONE broad genre by "
    "communicative FUNCTION (report/summarise/teach/worship/express/legislate), not topic. "
    "Key rule: a factual text that DESCRIBES or SUMMARISES a book is a book description and belongs to "
    "Informative, even if the book's subject is religion, politics or science; only scripture, worship "
    "and religious interpretation itself is Religious.\n"
    f"The 6 broad genres:\n{BLOCK}\n\nAnswer with ONLY the broad genre name, exactly as written."
)


cl = OpenAI()


def classify(text):
    """Return the broad genre the model assigns to a single text."""
    for _ in range(3):
        try:
            r = cl.chat.completions.create(
                model="gpt-5.6-luna",
                reasoning_effort="low",
                max_completion_tokens=300,
                messages=[
                    {"role": "system", "content": SYS},
                    {
                        "role": "user",
                        "content": f"Arabic text:\n{text[:1500]}\n\nBroad genre:",
                    },
                ],
            )
            o = r.choices[0].message.content or ""
            for b in NAMES:
                if b.lower() in o.lower():
                    return b
        except Exception:
            continue
    return None


def main():
    """Run the broad-only-plus-instruction arm over the verification pool."""
    txt = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    pool = json.loads(POOL.read_text())
    print(f"[arm-B] broad-only + anti-trap instruction, pool {len(pool)}", flush=True)
    res, done = {}, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, b in ex.map(lambda i: (i, classify(txt[i])), pool):
            res[i] = b
            done += 1
            if done % 500 == 0:
                print(f"[arm-B] {done}/{len(pool)}", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[arm-B] wrote {OUT}")


if __name__ == "__main__":
    main()
