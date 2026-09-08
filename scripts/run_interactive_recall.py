"""Run the targeted Interactive-recall pass used in the submitted base pipeline.

The broad gate under-predicted Interactive texts. This pass examines short texts near the decision
boundary and reassigns those it identifies as posts, comments, or reviews. Its prompt favors recall
while retaining lyrics, scripture, news, book descriptions, textbooks, job ads, and legal text
outside Interactive.
"""
import json, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test" / "test.json"
DEFS = ROOT / "test" / "test_genre_definitions.json"
CUR = Path(os.environ.get("INPUT_PREDICTIONS", ROOT / "work" / "pre_interactive_predictions.json"))
OUT = Path(os.environ.get("OUT", ROOT / "work" / "interactive_recall_predictions.json"))
WORKERS = int(os.environ.get("WORKERS", "64"))
client = OpenAI(
    base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("JUDGE_API_KEY", "local-vllm"),
    timeout=45.0,
    max_retries=0,
)
MODEL_ID = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")

EMO = re.compile(r"[\U0001F300-\U0001FAFF☀-➿❤♥♡\U0001F900-\U0001F9FF]"); HASH = re.compile(r"#[^\s#]{2,}")
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
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL_ID, temperature=0.0, max_tokens=40, extra_body={"seed": 42},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Arabic text:\n{text[:1500]}\n\nOutput the JSON."}])
            t = r.choices[0].message.content
            m = re.findall(r"\{[^{}]*\}", t)
            if m:
                o = json.loads(m[-1])
                return bool(o.get("interactive")), o.get("type")
        except Exception as exc:
            if attempt == 2:
                print(f"[interactive] request failed after 3 attempts: {exc}", file=sys.stderr)
    return None, None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s2b = {d["specific_genre"]: d["broad_genre"] for d in json.loads(DEFS.read_text())}
    rows = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    cur = {x["id"]: dict(x) for x in json.loads(CUR.read_text())}
    ids = [r["id"] for r in json.loads(TEST.read_text())]

    def border(i):
        b = cur[i]["broad_genre"]; t = rows[i]
        if b == "Interactive" or cur[i]["specific_genre"] in ("quran", "hadith") or len(t) > 400:
            return False
        return b in ("Creative", "Informative", "Learning", "Religious") and \
            (len(t) < 220 or EMO.search(t) or HASH.search(t) or SECOND.search(t))

    bl = [i for i in ids if border(i)]
    print(f"[intjudge] {len(bl)} borderline", flush=True)
    valid = {"twitter_posts", "youtube_comments", "app_reviews", "book_reviews"}
    flips = 0; done = 0

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
