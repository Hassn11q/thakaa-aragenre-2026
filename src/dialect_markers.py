"""Dialect cue lists for the five song-lyric classes, and the coverage measurement they support.

The paper reports that 87-99% of the texts we assign to a song-lyric class carry no marker
from that dialect's own cue list. The lists below are the ones that measurement uses; they
are lexical markers that survive in writing, which is exactly the point being made: the
features separating these dialects are largely phonetic and do not reach the page.

Run:  python3 src/dialect_markers.py --preds submissions/final_submission.json
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AB = "ء-ي"


def _wb(words):
    """Match any of `words` as a whole Arabic token."""
    return re.compile(rf"(?<![{AB}])(?:{'|'.join(words)})(?![{AB}])")


MARKERS = {
    "egyptian_song_lyrics": _wb(
        ["ده", "دي", "دى", "مش", "عايز", "عاوز", "ازاي", "علشان", "كده", "بتاع", "اهو", "خالص", "دلوقتي"]
    ),
    "gulf_song_lyrics": _wb(
        ["وايد", "شلونك", "چذي", "أبغى", "ابغى", "مب", "هاللي", "زين", "عساك", "يبه", "حيل", "طال عمرك"]
    ),
    "iraqi_song_lyrics": _wb(
        ["شكو", "ماكو", "هسه", "هسة", "اكو", "شلونچ", "خوش", "هواي", "تكول", "اكول", "چان", "وين رايح"]
    ),
    "levantine_song_lyrics": _wb(
        ["هيك", "شو", "بدي", "بدك", "هلق", "منيح", "كتير", "عم", "لسا", "شوي", "كرمال", "تعا"]
    ),
    "sudanese_song_lyrics": _wb(
        ["داير", "شنو", "زول", "ياخ", "كيف", "براك", "شديد", "كتير خالص", "جادي"]
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.json")
    ap.add_argument("--preds", type=Path, default=ROOT / "submissions" / "final_submission.json")
    a = ap.parse_args()

    if not a.test.exists():
        raise SystemExit(
            f"missing {a.test} (not redistributed; get it from the organisers). "
            "This table is computed over the test texts, so it needs them."
        )
    text = {r["id"]: (r.get("text") or "") for r in json.loads(a.test.read_text())}
    preds = json.loads(a.preds.read_text())

    for genre, rx in MARKERS.items():
        ids = [p["id"] for p in preds if p["specific_genre"] == genre]
        if not ids:
            continue
        bare = sum(1 for i in ids if not rx.search(text[i]))
        print(f"{genre:24s} n={len(ids):5d}  no marker {bare:5d}  ({100 * bare / len(ids):.0f}%)")


if __name__ == "__main__":
    main()
