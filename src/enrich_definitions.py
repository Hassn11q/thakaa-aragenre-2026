"""Build enriched genre definitions (ETP / LLM-refined-taxonomy) via free gemma.

For each of the 74 genres: original definition + an explicit contrast against its same-family
siblings + 2 short synthetic Arabic exemplar texts in that genre's register/dialect/domain. These
sharpen both the embedder (retrieval prototypes) and the judge (few-shot anchors). Genre-general:
built from the released definitions + Arabic knowledge, no dataset labels.
"""

import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
# Point JUDGE_URL at your own OpenAI-compatible endpoint; JUDGE_MODEL names the served model.
JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:9224/v1")
DEFS = ROOT / "data" / "test_genre_definitions.json"
OUT = ROOT / "work" / "enriched_definitions.json"
client = OpenAI(base_url="http://0.0.0.0:9224/v1", api_key="local", timeout=60.0, max_retries=0)
LLM = os.environ.get("JUDGE_MODEL", "/gemma-4-31b")


def main():
    """Generate contrastive descriptions for every genre and cache them."""
    d = json.loads(DEFS.read_text())
    fam = defaultdict(list)
    for x in d:
        fam[x["broad_genre"]].append(x)
    by_spec = {x["specific_genre"]: x for x in d}

    def enrich(x):
        """Return the contrastive description for one genre."""
        sibs = [
            s["specific_genre"]
            for s in fam[x["broad_genre"]]
            if s["specific_genre"] != x["specific_genre"]
        ]
        sib_defs = "\n".join(f"- {s}: {by_spec[s]['specific_genre_definition']}" for s in sibs[:12])
        prompt = (
            f"Genre: {x['specific_genre']}\nDefinition: {x['specific_genre_definition']}\n\n"
            f"Same-family sibling genres (must be told apart from these):\n{sib_defs}\n\n"
            "Task: (1) In ONE sentence, state the concrete features that distinguish THIS genre from its "
            "siblings (surface markers, dialect, register, subject, text-type). (2) Write 2 SHORT, realistic "
            "Arabic example snippets (each <=25 words) that are unambiguously THIS genre. "
            'Output ONLY JSON: {"contrast": "...", "examples": ["...", "..."]}.'
        )
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=LLM,
                    temperature=0.3,
                    max_tokens=400,
                    extra_body={"seed": 42},
                    messages=[{"role": "user", "content": prompt}],
                )
                m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
                if m:
                    o = json.loads(m.group())
                    return (
                        x["specific_genre"],
                        o.get("contrast", ""),
                        o.get("examples", [])[:2],
                    )
            except Exception:
                continue
        return x["specific_genre"], "", []

    enriched = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        for spec, contrast, examples in ex.map(enrich, d):
            enriched[spec] = {"contrast": contrast, "examples": examples}
            print(f"[enrich] {spec}: {len(examples)} examples", flush=True)

    out = []
    for x in d:
        e = enriched.get(x["specific_genre"], {})
        rich = x["specific_genre_definition"]
        if e.get("contrast"):
            rich += f" | يتميّز عن أقاربه: {e['contrast']}"
        if e.get("examples"):
            rich += " | أمثلة: " + " /// ".join(e["examples"])
        out.append({**x, "enriched_definition": rich})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[enrich] wrote {OUT}")


if __name__ == "__main__":
    main()
