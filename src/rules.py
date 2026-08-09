"""Expert-linguist high-precision rule layer, applied AFTER the LLM judge.

Some genres are diagnosable from surface orthography/format with near-100% precision — exactly
what a human Arabic expert keys on. We override the judge ONLY when such a marker fires, which
fixes both the broad gate and the specific label (e.g. mushaf orthography => quran; isnad opening
=> hadith). Everything else keeps the judge's prediction.

Usage:
  python src/rules.py --pred work/judge_predictions.json --out work/judge_predictions_ruled.json
  python src/rules.py --pred <p> --measure     # report per-rule coverage + how many it flips
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"

QMARK = re.compile(r"[ۖ-ۜ۟-ۤ۩ٱ]")  # Quran mushaf pause marks + wasla + small high marks
HADITH_START = re.compile(r"^\s*(حد[َّ]?ث[َ]?ن[َا|ي]|حَدَّثَنَا|أخبرنا|أَخْبَرَنَا|حدثني)")

# Interactive-recall rule (FIX_BROAD_PLAYBOOK Q2): the gate ~2x under-predicts Interactive; short
# emoji/hashtag/opinion comments get locked into Creative(song)/Informative(blurb). Reassign only on
# high-precision surface markers. Arabic-boundary anchors (not \b) so praise words don't match inside
# bigger words; bare "جدا" deliberately omitted (matches inside مجدا / bible text).
EMO = re.compile(r"[\U0001F300-\U0001FAFF☀-➿❤♥♡\U0001F900-\U0001F9FF]")
HASH = re.compile(r"#[^\s#]{2,}")
AB = "ء-ي"  # Arabic letters
OPIN = re.compile(
    rf"(?<![{AB}])(?:رائع|روعة|ممل|سيء|سيئ|جميل|أنصح|ما شاء الله|تحفة|مبدع|فنان|رهيب|زفت|"
    rf"أفضل|أحسن|يستحق|لا يمل|أجمل|احلى|أحلى|حلو)(?![{AB}])|[👏❤️❤🔥]"
)
YT_CUE = re.compile(r"الحلقة|الحلقه|الفيديو|القناة|القناه|بودكاست|فيديو|مقطع|البودكاست")
APP_CUE = re.compile(r"(?:التطبيق|تطبيق|البرنامج|التطبيقات)")
BOOK_CUE = re.compile(r"(?:الكتاب|الرواية|رواية|الكاتب|الكاتبة|كتاب)")


def interactive_rule(t):
    """Return an Interactive (broad, specific) if high-precision markers fire, else None."""
    if len(t) > 200:
        return None
    has_hash = bool(HASH.search(t))
    has_emo = bool(EMO.search(t))
    has_opin = bool(OPIN.search(t))
    wc = len(t.split())
    fire = (
        has_hash
        or (len(t) <= 130 and has_emo and has_opin)
        or (len(t) <= 90 and has_opin and wc <= 14)
    )
    if not fire:
        return None
    if has_hash or "@" in t:
        spec = "twitter_posts"
    elif YT_CUE.search(t):
        spec = "youtube_comments"
    elif APP_CUE.search(t) and has_opin:
        spec = "app_reviews"
    elif BOOK_CUE.search(t) and has_opin:
        spec = "book_reviews"
    else:
        spec = "youtube_comments"
    return ("Interactive", spec)


def rules(text):
    """Return (broad, specific) to force, or None. Ordered by precision; first match wins."""
    t = text or ""
    if len(QMARK.findall(t)) >= 2:
        return ("Religious", "quran")
    if HADITH_START.match(t):
        return ("Religious", "hadith")
    ir = interactive_rule(t)
    if ir:
        return ir
    return None


def main():
    """Apply the surface rules to a prediction file and report what changed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args()

    text = {r["id"]: (r.get("text") or "") for r in json.loads(TEST.read_text())}
    SPEC_TO_BROAD = {
        d["specific_genre"]: d["broad_genre"] for d in json.loads(DEFS.read_text())
    }
    preds = json.loads(a.pred.read_text())

    fired = flipped = 0
    flips_by = Counter()
    out = []
    for p in preds:
        r = rules(text.get(p["id"], ""))
        if r:
            fired += 1
            b, s = r
            if p["specific_genre"] != s:
                flipped += 1
                flips_by[f"{p['specific_genre']} -> {s}"] += 1
            p = {"id": p["id"], "broad_genre": SPEC_TO_BROAD[s], "specific_genre": s}
        out.append(p)

    print(f"[rules] fired on {fired} rows, flipped {flipped} (vs input pred)")
    for k, n in flips_by.most_common(12):
        print(f"   {k}: {n}")
    if a.measure:
        return
    outp = a.out or a.pred.with_name(a.pred.stem + "_ruled.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[rules] wrote -> {outp}")


if __name__ == "__main__":
    main()
