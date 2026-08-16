"""Phase B (v3): hierarchical family-restricted specific judge.

Broad is already reliable (leaderboard broad macro 0.84 / acc 0.88), so gate on the v2 broad
prediction and choose the specific genre ONLY among that broad family's siblings:
  - small families: show every sibling definition
    Legal 3) -> show ALL siblings, so the correct label can never be missed by retrieval;
  - large families: narrow with the retrieval ranking and keep the top TOP_K in-family.
Then one setwise topic-aware gemma call scores the candidates; marginal calibration + argmax.

This removes cross-family distractors (the main residual error) and eliminates within-family
retrieval recall loss for the small, hard families (song dialects, textbook levels, quran/tafsir).
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
# Point JUDGE_URL at your own OpenAI-compatible endpoint; JUDGE_MODEL names the served model.
JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:9224/v1")
# BOOTSTRAP=1 judges across all families to create the broad gate from scratch.
BOOTSTRAP = os.environ.get("BOOTSTRAP", "0") == "1"
BOOTSTRAP_TOP_K = int(os.environ.get("BOOTSTRAP_TOP_K", "8"))
CANDIDATE_GLOB = "candidates_*of*.json"
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"
BROAD_GATE = Path(
    os.environ.get("BROAD_GATE", ROOT / "artifacts" / "broad_gate.json")
)  # family gate
FALLBACK_PREDICTIONS = Path(
    os.environ.get("FALLBACK_PREDICTIONS", ROOT / "artifacts" / "broad_gate.json")
)  # specific used only when a big family filters empty
OUT = Path(os.environ.get("OUT", ROOT / "work" / "judge_predictions.json"))
SCORES = ROOT / "work" / "judge_scores.json"
ALPHA = float(os.environ.get("ALPHA", "0.75"))
TEXT_CAP = 6000
WORKERS = int(os.environ.get("WORKERS", "24"))

client = OpenAI(
    base_url=JUDGE_URL,
    api_key=os.environ.get("JUDGE_API_KEY", "local"),
    timeout=45.0,
    max_retries=0,
)
LLM = os.environ.get("JUDGE_MODEL", "/gemma-4-31b")
SYS = (
    "You are a world-class Arabic philologist and corpus linguist. Given an Arabic text and a "
    "numbered list of candidate genre DEFINITIONS from the SAME broad family, identify the ONE genre "
    "the text actually belongs to. Work like an expert: first find the single MOST DIAGNOSTIC signal "
    "in the text — a dialect marker (Egyptian ده/عايز, Gulf وايد/شلونك, Iraqi شكو/تكول, Levantine "
    "هيك/بدي, Sudanese داير/زول), Quranic mushaf orthography vs an isnad chain (حدثنا…عن) vs verse "
    "commentary (قوله تعالى), a news dateline (مدينة أ.ف.ب:), meta book-blurb language describing a "
    "book (الكتاب/المؤلف/يتناول), a job-ad frame (مطلوب…خبرة) and its DOMAIN, a numbered legal/UN "
    "clause, or a Christian marker (يسوع/المسيح/الإنجيل) — then decide by SUBJECT + DIALECT/VARIETY + "
    "TEXT-TYPE + REGISTER together. Rules: a text that merely mentions a topic is NOT that topic's "
    "genre unless its FORM matches (e.g. a blurb ABOUT an Islamic book is a book description, not "
    "scripture); pick the exact domain/dialect, never a near-sibling. Score DECISIVELY and "
    "EXCLUSIVELY: give the single best-matching definition ~90-100 and clearly wrong ones near 0; "
    "avoid ties. For EACH candidate output an integer 0-100. Output ONLY a JSON array of integers, "
    "one per candidate, in the given order."
)

# Family-specific discriminative marker cues (SOTA #5): the dialect and religious-text-type
# distinctions are lexical/orthographic and invisible to definition semantics alone. These are
# given to the judge as "look-for" signals. Sources: NADI-2024 dialect ID; Arabic religious corpora.
CUES = {
    "Creative": (
        "DISCRIMINATIVE CUES (dialect / form): Egyptian lyrics: ده دي دى مش عايز عاوز ازاي علشان انت "
        "كده; Gulf: وايد شلونك چذي أبغى مب اللي هاللي زين; Iraqi: شكو ماكو هسه اكو شلونچ خوش هواي; "
        "Levantine: هيك شو بدي هلق منيح كتير عم; Sudanese: داير كيف شنو زول ياخ; msa_poetry / "
        "classical_poetry: NO colloquial dialect, formal fuṣḥā, metre and rhyme, elevated vocabulary "
        "(classical = older/heritage diction, msa = modern standard)."
    ),
    "Religious": (
        "DISCRIMINATIVE CUES (text type): quran: fully diacritized divine speech, آية/سورة, no isnad; "
        "hadith: chains of transmission — حدثنا/أخبرنا … عن فلان عن فلان، قال رسول الله ﷺ; tafsir: "
        "verse commentary — قوله تعالى، أي، يعني، المعنى، قال المفسرون; sharia_law_book_description / "
        "islamic_book_description: scholarly Islamic prose ABOUT religion; bible: Christian scripture "
        "narrative; biblical: Christian commentary/interpretation; coptic_devotional_literature: "
        "Coptic-Christian prayer/liturgy."
    ),
    "Learning": (
        "DISCRIMINATIVE CUES (level / form): early_education / upper_elementary_textbooks: very simple "
        "short sentences, basic vocabulary; intermediate / high_school_textbooks: denser academic "
        "vocabulary and structured lessons; *_student_writing: first-person learner essays grouped by "
        "topic (sports, culture, technology, values, communication, national development, schoolwork)."
    ),
}


FAILURES = []  # scoring calls that exhausted their retries


def setwise_scores(text, defs, k, cue=""):
    """Score all candidate definitions for one text in a single request."""
    cot = os.environ.get("COT", "0") == "1"
    block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(defs))
    cue_block = f"\n\n{cue}" if cue else ""
    if cot:
        user = (
            f"Arabic text:\n{text}\n\nCandidate genre definitions:\n{block}{cue_block}\n\n"
            f"Reason step by step in <=40 words about the text's diagnostic features (subject, "
            f"dialect/variety, register, text-type), then on the FINAL line output a JSON array of "
            f"{k} integers (0-100), scoring candidates 1..{k} in order. Final line = the array only."
        )
        mx = 400
    else:
        user = (
            f"Arabic text:\n{text}\n\nCandidate genre definitions:\n{block}{cue_block}\n\n"
            f"Output a JSON array of {k} integers (0-100), scoring candidates 1..{k} in order."
        )
        mx = 96
    txt = None
    last_error = None
    for _ in range(3):  # retry transient errors under high concurrency
        try:
            r = client.chat.completions.create(
                model=LLM,
                temperature=0.0,
                max_tokens=mx,
                extra_body={"seed": 42},
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": user},
                ],
            )
            txt = r.choices[0].message.content.strip()
            break
        except Exception as exc:
            last_error = exc
            continue
    if txt is None:
        FAILURES.append(last_error)
        return None
    arrs = re.findall(r"\[[^\[\]]*\]", txt)  # CoT: scores are the LAST array; plain: the only one
    for cand in reversed(arrs):
        try:
            arr = [float(x) for x in json.loads(cand)][:k]
            if len(arr) == k:
                return np.array(arr, dtype=float)
        except Exception:
            continue
    nums = re.findall(r"\d+", txt)
    if len(nums) >= k:
        return np.array([float(x) for x in nums[-k:]])  # last k numbers = the score line
    return None


def main():
    """Score every text against its family-restricted candidates and write predictions."""
    defrows = json.loads(DEFS.read_text())
    enrich = os.environ.get("ENRICH", "1") == "1"
    enriched_file = ROOT / "work" / "enriched_definitions.json"
    if os.environ.get("ENRICHED_DEFS") == "1" and enriched_file.exists():
        defmap = {
            d["specific_genre"]: d["enriched_definition"]
            for d in json.loads(enriched_file.read_text())
        }
    elif enrich:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from discriminators import DISCRIMINATORS

        defmap = {
            d["specific_genre"]: d["specific_genre_definition"]
            + (
                f" | كيف تميّزه: {DISCRIMINATORS[d['specific_genre']]}"
                if d["specific_genre"] in DISCRIMINATORS
                else ""
            )
            for d in defrows
        }
    else:
        defmap = {d["specific_genre"]: d["specific_genre_definition"] for d in defrows}
    SPEC_TO_BROAD = {d["specific_genre"]: d["broad_genre"] for d in defrows}
    fam = defaultdict(list)
    for d in defrows:
        fam[d["broad_genre"]].append(d["specific_genre"])

    if BOOTSTRAP:
        # No gate yet: judge over the retrieval ranking across all families and let the
        # chosen specific genre define the broad family. This is how broad_gate.json is
        # produced when starting from nothing.
        broad_by_id, spec_fallback_by_id = {}, {}
    else:
        broad_by_id = {x["id"]: x["broad_genre"] for x in json.loads(BROAD_GATE.read_text())}
        spec_fallback_by_id = {
            x["id"]: x["specific_genre"]
            for x in json.loads(FALLBACK_PREDICTIONS.read_text())
        }
    text_by_id = {r["id"]: (r.get("text") or "")[:TEXT_CAP] for r in json.loads(TEST.read_text())}

    top20 = {}
    for f in sorted((ROOT / "work").glob(CANDIDATE_GLOB)):
        for it in json.loads(f.read_text())["topk"]:
            top20[it["id"]] = it["top_genres"]
    ids = [r["id"] for r in json.loads(TEST.read_text())]

    # v7: tight retrieval-pruned candidates. Enumerating whole families (v5) diluted the judge and
    # regressed (0.6517 vs v3's retrieval-pruned 0.6882). Papers agree: family ∩ retrieval-top-k wins.
    # DIALECT/TEXT-TYPE families (Creative songs, Religious) are NOT separable by retrieval over English
    # definitions — the judge reads the Arabic and decides, so show ALL (they are small, <=7). Small
    # Interactive/Legal: show all. TOPIC families (Informative, Learning): keep only the top-TOP_K
    # retrieved in-family (floor handled by availability).
    SHOW_ALL = {"Creative", "Religious", "Interactive", "Legal"}
    TOP_K = int(os.environ.get("TOP_K", "6"))

    def candidates(rid):
        """Return the candidate genres allowed for one text."""
        if BOOTSTRAP:
            return top20.get(rid, [])[:BOOTSTRAP_TOP_K]
        b = broad_by_id[rid]
        sibs = fam[b]
        if b in SHOW_ALL:
            return list(sibs)
        infam = [s for s in top20.get(rid, []) if s in set(sibs)]  # retrieval-ranked, in-family
        if not infam:
            return [spec_fallback_by_id[rid]]
        return infam[:TOP_K]

    PASSES = int(
        os.environ.get("PASSES", "1")
    )  # permutation self-consistency: rotate candidate order

    def work(rid):
        """Score and label a single text."""
        cands = candidates(rid)
        k = len(cands)
        if k == 1:
            return cands, np.array([1.0])
        cue = CUES.get(broad_by_id.get(rid, ""), "") if not BOOTSTRAP else ""
        defs = [defmap[s] for s in cands]
        acc = np.zeros(k, dtype=float)
        n = 0
        for p in range(PASSES):
            rot = p % k  # deterministic rotation as the permutation
            order = list(range(rot, k)) + list(range(rot))
            sc = setwise_scores(text_by_id.get(rid, ""), [defs[o] for o in order], k, cue)
            if sc is None:
                continue
            inv = np.zeros(k)
            for pos, o in enumerate(order):
                inv[o] = sc[pos]
            acc += inv
            n += 1
        if n == 0:
            acc = np.array([k - i for i in range(k)], dtype=float)
        else:
            acc /= n
        return cands, acc

    per = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rid, res in zip(ids, ex.map(work, ids)):
            per[rid] = res
            done += 1
            if done % 2000 == 0:
                print(f"[hier] judged {done}/{len(ids)}", flush=True)

    tot, cnt = {}, {}
    for g, sc in per.values():
        for gi, v in zip(g, sc):
            tot[gi] = tot.get(gi, 0.0) + v
            cnt[gi] = cnt.get(gi, 0) + 1
    prior = {gi: tot[gi] / cnt[gi] for gi in tot}

    preds = []
    for rid in ids:
        g, sc = per[rid]
        adj = np.array([sc[i] - ALPHA * prior.get(g[i], 0.0) for i in range(len(g))])
        spec = g[int(adj.argmax())]
        preds.append({"id": rid, "broad_genre": SPEC_TO_BROAD[spec], "specific_genre": spec})

    if FAILURES:
        rate = len(FAILURES) / max(len(ids), 1)
        print(
            f"[hier] WARNING: {len(FAILURES)} of {len(ids)} texts could not be scored "
            f"({rate:.1%}); those fell back to retrieval order. Last error: {FAILURES[-1]}",
            file=sys.stderr,
        )
        if rate > 0.05:
            raise SystemExit(
                f"refusing to write {OUT}: {rate:.1%} of texts were never scored, so the "
                "output would mostly be retrieval order rather than judge decisions"
            )

    OUT.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    SCORES.write_text(
        json.dumps(
            [
                {
                    "id": rid,
                    "genres": per[rid][0],
                    "scores": [round(float(x), 1) for x in per[rid][1]],
                }
                for rid in ids
            ],
            ensure_ascii=False,
        )
    )
    print(f"[hier] wrote {len(preds)} -> {OUT}; ALPHA={ALPHA}")
    print(
        "[hier] specific top12:",
        Counter(p["specific_genre"] for p in preds).most_common(12),
    )
    print("[hier] broad:", Counter(p["broad_genre"] for p in preds).most_common())


if __name__ == "__main__":
    main()
