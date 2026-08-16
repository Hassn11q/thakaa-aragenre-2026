"""Targeted high-recall Interactive judge pass, applied during the evaluation phase.

This is the reassignment step that artifacts/base_predictions.json carries on top of a plain
src/judge.py run; it is released for inspection but is not part of reproduce.sh, which rebuilds
the submission from the cached base predictions.

Interactive is ~2x under-predicted; the surface rule caught emoji/hashtag comments, but dialectal
comments WITHOUT emoji stay locked as song lyrics. This binary judge runs ONLY on borderline short
texts and reassigns confirmed social posts / comments / reviews to Interactive. Biased slightly to
YES because Interactive is under-predicted, but real lyrics/scripture/news must stay NO.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"
CUR = Path(os.environ.get("CUR", ROOT / "work" / "judge_predictions.json"))
OUT = Path(os.environ.get("OUT", ROOT / "work" / "interactive_judged.json"))
WORKERS = int(os.environ.get("WORKERS", "64"))
JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:9224/v1")
client = OpenAI(base_url=JUDGE_URL, api_key="local", timeout=45.0, max_retries=0)
LLM = os.environ.get("JUDGE_MODEL", "/gemma-4-31b")

EMO = re.compile(r"[\U0001F300-\U0001FAFF☀-➿❤♥♡\U0001F900-\U0001F9FF]")
HASH = re.compile(r"#[^\s#]{2,}")
SECOND = re.compile(r"(?<![ء-ي])(?:انت|انتي|إنت|أنت|يا|ياخي|ياخوي|والله|ليش|شنو|ايش|وش)(?![ء-ي])")

SYS = ("You decide if an Arabic text is an INTERACTIVE online text written BY a user to address others "
       "or react to content: a social-media post (twitter_posts), a comment on a video/podcast/channel "
       "(youtube_comments), a user review of an app/software (app_reviews), or a user review of a book "
       "(book_reviews). It is NOT interactive if it is actual song lyrics or poetry, Quran/hadith/"
       "scripture, a news report, an encyclopedia entry, a book blurb/description, a textbook, a job ad, "
       "or a legal text. Guidance: short first-person reactions, praise or criticism of something, direct "
       "address to a person/creator, hashtags, or opinions ABOUT content are usually interactive; a "
       "poem/song expresses emotion as art (not a reaction to content). Output ONLY a JSON object: "
       '{"interactive": true or false, "type": "twitter_posts" | "youtube_comments" | "app_reviews" | '
       '"book_reviews" | null}.')


def judge(text):
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=LLM, temperature=0.0, max_tokens=40, extra_body={"seed": 42},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Arabic text:\n{text[:1500]}\n\nOutput the JSON."}])
            t = r.choices[0].message.content
            m = re.findall(r"\{[^{}]*\}", t)
            if m:
                o = json.loads(m[-1])
                return bool(o.get("interactive")), o.get("type")
        except Exception:
            continue
    return None, None


def main():
    s2b = {d["specific_genre"]: d["broad_genre"] for d in json.loads(DEFS.read_text())}
    rows = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    cur = {x["id"]: dict(x) for x in json.loads(CUR.read_text())}
    ids = [r["id"] for r in json.loads(TEST.read_text())]

    def border(i):
        b = cur[i]["broad_genre"]
        t = rows[i]
        if b == "Interactive" or cur[i]["specific_genre"] in ("quran", "hadith") or len(t) > 400:
            return False
        return b in ("Creative", "Informative", "Learning", "Religious") and \
            (len(t) < 220 or EMO.search(t) or HASH.search(t) or SECOND.search(t))

    bl = [i for i in ids if border(i)]
    print(f"[intjudge] {len(bl)} borderline", flush=True)
    valid = {"twitter_posts", "youtube_comments", "app_reviews", "book_reviews"}
    flips = 0
    done = 0

    def work(i):
        return i, judge(rows[i])
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (is_int, typ) in ex.map(work, bl):
            done += 1
            if is_int:
                spec = typ if typ in valid else "youtube_comments"
                cur[i] = {"id": i, "broad_genre": "Interactive", "specific_genre": spec}
                flips += 1
            if done % 1500 == 0:
                print(f"[intjudge] {done}/{len(bl)} flips={flips}", flush=True)

    out = [{"id": i, "broad_genre": s2b[cur[i]["specific_genre"]], "specific_genre": cur[i]["specific_genre"]} for i in ids]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    from collections import Counter
    print(f"[intjudge] flips->Interactive {flips}; wrote {OUT}")
    print("[intjudge] broad:", Counter(p["broad_genre"] for p in out).most_common())


if __name__ == "__main__":
    main()
