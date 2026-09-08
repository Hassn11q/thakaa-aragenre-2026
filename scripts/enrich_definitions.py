"""Build the enriched genre definitions used by the retrieval and judge stages.

For each genre, Gemma 4 31B IT adds a contrast with same-family labels and two short synthetic Arabic
examples. The inputs are the released definitions and Arabic-language cues; dataset labels are not
used.
"""
import json, os, re, sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFS = ROOT / "test" / "test_genre_definitions.json"
OUT = ROOT / "work" / "enriched_definitions.json"
client = OpenAI(
    base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("JUDGE_API_KEY", "local-vllm"),
    timeout=60.0,
    max_retries=0,
)
MODEL_ID = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(DEFS.read_text())
    fam = defaultdict(list)
    for x in d:
        fam[x["broad_genre"]].append(x)
    by_spec = {x["specific_genre"]: x for x in d}

    def enrich(x):
        sibs = [s["specific_genre"] for s in fam[x["broad_genre"]] if s["specific_genre"] != x["specific_genre"]]
        sib_defs = "\n".join(f"- {s}: {by_spec[s]['specific_genre_definition']}" for s in sibs[:12])
        prompt = (
            f"Genre: {x['specific_genre']}\nDefinition: {x['specific_genre_definition']}\n\n"
            f"Same-family sibling genres (must be told apart from these):\n{sib_defs}\n\n"
            "Task: (1) In ONE sentence, state the concrete features that distinguish THIS genre from its "
            "siblings (surface markers, dialect, register, subject, text-type). (2) Write 2 SHORT, realistic "
            "Arabic example snippets (each <=25 words) that are unambiguously THIS genre. "
            'Output ONLY JSON: {"contrast": "...", "examples": ["...", "..."]}.')
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL_ID, temperature=0.3, max_tokens=400, extra_body={"seed": 42},
                    messages=[{"role": "user", "content": prompt}])
                m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
                if m:
                    o = json.loads(m.group())
                    return x["specific_genre"], o.get("contrast", ""), o.get("examples", [])[:2]
            except Exception as exc:
                if attempt == 2:
                    print(f"[enrich] {x['specific_genre']} failed after 3 attempts: {exc}", file=sys.stderr)
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
