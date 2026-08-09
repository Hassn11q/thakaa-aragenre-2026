"""Full-context definition-guided SPECIFIC prediction (trap-immune) via gemini or gpt.

Feeds the ENTIRE taxonomy (all 74 specific definitions grouped by broad) + the text, asks for the
single best specific_genre. Broad is derived as parent(specific). Because the model sees that e.g.
islamic_book_description is an Informative book-summary, it will not fall for the topic trap
(text about Islam -> Religious). PROVIDER=gemini|gpt.
"""

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"
POOL = Path(os.environ.get("POOL", ROOT / "work" / "broad_pool.json"))
PROVIDER = os.environ.get("PROVIDER", "gpt")
OUT = Path(os.environ.get("OUT", ROOT / "work" / f"verify_{PROVIDER}.json"))
WORKERS = int(os.environ.get("WORKERS", "20"))

defs = json.loads(DEFS.read_text())
SPECS = [d["specific_genre"] for d in defs]
SPEC_TO_BROAD = {d["specific_genre"]: d["broad_genre"] for d in defs}
fam = defaultdict(list)
for d in defs:
    fam[d["broad_genre"]].append(d)
TAX = "\n".join(
    f"## {b}\n"
    + "\n".join(f"- {x['specific_genre']}: {x['specific_genre_definition']}" for x in fam[b])
    for b in fam
)
SYS = (
    "You are an expert Arabic text-genre classifier. Given an Arabic text and the FULL taxonomy of 74 "
    "specific genres (grouped by their 6 broad genres), pick the ONE specific_genre that best fits by "
    "communicative FUNCTION and text-type, NOT topic alone. Key rule: a factual text that DESCRIBES or "
    "SUMMARISES a book (e.g. 'this book presents...') is a *_book_description (Informative), even if the "
    "book's topic is religion/politics/science. Scripture/worship/interpretation itself is Religious. "
    "Reply with ONLY the exact specific_genre identifier.\n\nTAXONOMY:\n" + TAX
)


def make_call():
    """Return a function that sends one prompt to the configured provider."""
    if PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        cl = genai.Client()

        def call(text):
            """Send one prompt to the provider and return the raw reply."""
            r = cl.models.generate_content(
                model="gemini-3.6-flash",
                contents=SYS + f"\n\nArabic text:\n{text[:1500]}\n\nBest specific_genre:",
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=400),
            )
            return r.text or ""

        return call
    else:
        from openai import OpenAI

        cl = OpenAI()

        def call(text):
            """Send one prompt to the provider and return the raw reply."""
            r = cl.chat.completions.create(
                model="gpt-5.6-luna",
                messages=[
                    {"role": "system", "content": SYS},
                    {
                        "role": "user",
                        "content": f"Arabic text:\n{text[:1500]}\n\nBest specific_genre:",
                    },
                ],
                reasoning_effort="low",
                max_completion_tokens=500,
            )
            return r.choices[0].message.content or ""

        return call


def main():
    """Re-predict the pooled instances with one provider and cache the verdicts."""
    call = make_call()
    txt = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    pool = json.loads(POOL.read_text())
    print(f"[verify-{PROVIDER}] pool {len(pool)}", flush=True)

    def classify(text):
        """Return the specific genre the model picks for one text."""
        for _ in range(3):
            try:
                t = call(text)
                hits = [g for g in SPECS if g in t]
                if hits:
                    return max(hits, key=len)  # longest match = most specific
            except Exception:
                continue
        return None

    res = {}
    done = 0

    def work(i):
        """Classify a single instance and return it with its id."""
        return i, classify(txt[i])

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, g in ex.map(work, pool):
            res[i] = g
            done += 1
            if done % 200 == 0:
                print(f"[verify-{PROVIDER}] {done}/{len(pool)}", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[verify-{PROVIDER}] wrote {OUT}")


if __name__ == "__main__":
    main()
