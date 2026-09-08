"""Predict a specific genre with GPT 5.6 Luna or Gemini 3.6 Flash using the full taxonomy.

The prompt includes all 74 specific definitions, grouped by broad genre, and asks for one specific
label. The broad label is derived from its parent. Seeing book-description labels helps the
verifier distinguish topic from text type. Set ``PROVIDER`` to ``gemini`` or ``gpt``.
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test" / "test.json"
DEFS = ROOT / "test" / "test_genre_definitions.json"
POOL = Path(os.environ.get("POOL", ROOT / "work" / "verification_pool.json"))
PROVIDER = os.environ.get("PROVIDER", "gpt")
OUT = Path(os.environ.get("OUT", ROOT / "work" / f"{PROVIDER}_verifier_predictions.json"))
WORKERS = int(os.environ.get("WORKERS", "20"))
GPT_MODEL_ID = os.environ.get("GPT_MODEL_ID", "gpt-5.6-luna")
GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.6-flash")

defs = json.loads(DEFS.read_text())
SPECS = [d["specific_genre"] for d in defs]
s2b = {d["specific_genre"]: d["broad_genre"] for d in defs}
fam = defaultdict(list)
for d in defs:
    fam[d["broad_genre"]].append(d)
TAX = "\n".join(f"## {b}\n" + "\n".join(f"- {x['specific_genre']}: {x['specific_genre_definition']}" for x in fam[b]) for b in fam)
SYS = ("You are an expert Arabic text-genre classifier. Given an Arabic text and the FULL taxonomy of 74 "
       "specific genres (grouped by their 6 broad genres), pick the ONE specific_genre that best fits by "
       "communicative FUNCTION and text-type, NOT topic alone. Key rule: a factual text that DESCRIBES or "
       "SUMMARISES a book (e.g. 'this book presents...') is a *_book_description (Informative), even if the "
       "book's topic is religion/politics/science. Scripture/worship/interpretation itself is Religious. "
       "Reply with ONLY the exact specific_genre identifier.\n\nTAXONOMY:\n" + TAX)


def make_call():
    if PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        cl = genai.Client()
        def call(text):
            r = cl.models.generate_content(model=GEMINI_MODEL_ID,
                contents=SYS + f"\n\nArabic text:\n{text[:1500]}\n\nBest specific_genre:",
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=400))
            return r.text or ""
        return call
    else:
        from openai import OpenAI
        cl = OpenAI()
        def call(text):
            r = cl.chat.completions.create(model=GPT_MODEL_ID,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Arabic text:\n{text[:1500]}\n\nBest specific_genre:"}],
                reasoning_effort="low", max_completion_tokens=500)
            return r.choices[0].message.content or ""
        return call


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    call = make_call()
    txt = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    pool = json.loads(POOL.read_text())
    print(f"[fullctx-{PROVIDER}] pool {len(pool)}", flush=True)

    def classify(text):
        for attempt in range(3):
            try:
                t = call(text)
                hits = [g for g in SPECS if g in t]
                if hits:
                    return max(hits, key=len)  # longest match = most specific
            except Exception as exc:
                if attempt == 2:
                    print(f"[fullctx-{PROVIDER}] request failed after 3 attempts: {exc}", file=sys.stderr)
        return None

    res = {}
    done = 0
    def work(i):
        return i, classify(txt[i])
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, g in ex.map(work, pool):
            res[i] = g
            done += 1
            if done % 200 == 0:
                print(f"[fullctx-{PROVIDER}] {done}/{len(pool)}", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[fullctx-{PROVIDER}] wrote {OUT}")


if __name__ == "__main__":
    main()
