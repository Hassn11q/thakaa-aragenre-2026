"""Reproduce the broad-only prompting figures in the paper, offline.

Checks two claims about the 1,403 `*_book_description` instances in the trap pool:

  1. all three verifiers call 157 of them Religious unanimously (full coverage);
  2. requiring two verifiers to agree, joint Religious calls fall 134 -> 37 under the
     full taxonomy -- restricted to the 548 the second verifier also covers there,
     because the deployed run queries it only on first-verifier disagreements.

Needs no GPU and no API access; reads only files in artifacts/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(name):
    return json.loads((ART / name).read_text())


def main():
    defs = json.loads((ROOT / "data" / "test_genre_definitions.json").read_text())
    parent = {d["specific_genre"]: d["broad_genre"] for d in defs}

    base = {r["id"]: r for r in load("base_predictions.json")}
    pool = load("trap_pool.json")
    book = [i for i in pool if base[i]["specific_genre"].endswith("book_description")]

    gpt_b, gem_b, son_b = load("gpt_broad.json"), load("gemini_broad.json"), load("sonnet_broad.json")
    gpt_f, gem_f = load("verify_gpt.json"), load("verify_gemini.json")

    covered = [i for i in book if i in gpt_b and i in gem_b and i in son_b]
    unanimous = sum(gpt_b[i] == gem_b[i] == son_b[i] == "Religious" for i in covered)

    # matched coverage: the second verifier must have a call under BOTH prompts
    matched = [i for i in book if i in gem_b and i in gem_f]
    fam = lambda src, i: parent.get(src.get(i))
    joint_broad = sum(gpt_b.get(i) == "Religious" and gem_b.get(i) == "Religious" for i in matched)
    joint_full = sum(fam(gpt_f, i) == "Religious" and fam(gem_f, i) == "Religious" for i in matched)

    solo_broad = sum(gpt_b.get(i) == "Religious" for i in book)
    solo_full = sum(fam(gpt_f, i) == "Religious" for i in book)

    print(f"book descriptions in pool          {len(book)}")
    print(f"  with all three broad-only calls  {len(covered)}")
    print(f"  unanimous Religious              {unanimous}   (paper: 157)")
    print()
    print(f"matched-coverage subset            {len(matched)}   (paper: 548)")
    print(f"  two verifiers agree, broad-only  {joint_broad}   (paper: 134)")
    print(f"  two verifiers agree, full taxon. {joint_full}   (paper: 37)")
    if joint_broad:
        print(f"  reduction                        {100 * (joint_broad - joint_full) / joint_broad:.0f}%   (paper: 72%)")
    print()
    print(f"first verifier alone, all {len(book)}    {solo_broad} -> {solo_full}   (paper: 394 -> 126)")


if __name__ == "__main__":
    main()
