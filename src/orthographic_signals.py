"""Diacritic and emoji coverage per genre group, the measurement behind our no-preprocessing choice.

Standard Arabic normalisation dediacritises and strips emoji. Both operations delete features
this taxonomy leans on: diacritisation separates scripture and poetry from everything else, and
emoji separate Interactive posts from everything else. The rates below are properties of the
text, so they are measurable without gold labels.

Run:  python3 src/orthographic_signals.py
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARABIC = re.compile("[ء-ي]")
DIACRITIC = re.compile("[ً-ْٰٖ-ٟۖ-ۭ]")
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")

HEAVY = 0.10  # share of Arabic characters carrying a mark


def diacritic_density(text):
    """Fraction of Arabic characters in `text` that carry a diacritic."""
    letters = len(ARABIC.findall(text))
    return len(DIACRITIC.findall(text)) / letters if letters else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.json")
    ap.add_argument("--preds", type=Path, default=ROOT / "submissions" / "final_submission.json")
    a = ap.parse_args()

    text = {r["id"]: (r.get("text") or "") for r in json.loads(a.test.read_text())}
    preds = json.loads(a.preds.read_text())

    by_genre = {}
    for p in preds:
        by_genre.setdefault(p["specific_genre"], []).append(p["id"])

    book_desc = [g for g in by_genre if g.endswith("_book_description")]
    groups = {
        "quran": ["quran"],
        "poetry": ["classical_poetry", "msa_poetry"],
        "hadith": ["hadith"],
        "book descriptions": book_desc,
        "Interactive": [
            "youtube_comments",
            "facebook_comments",
            "twitter_posts",
            "instagram_comments",
        ],
    }

    print(f"{'group':20s}{'n':>7s}{'heavy diacritics':>18s}{'emoji':>8s}")
    for name, genres in groups.items():
        ids = [i for g in genres for i in by_genre.get(g, [])]
        if not ids:
            continue
        heavy = sum(1 for i in ids if diacritic_density(text[i]) > HEAVY)
        emoji = sum(1 for i in ids if EMOJI.search(text[i]))
        print(f"{name:20s}{len(ids):7d}{100 * heavy / len(ids):17.0f}%{100 * emoji / len(ids):7.0f}%")


if __name__ == "__main__":
    main()
