"""Apply test-time prior correction (EMQ/SLD) to saved judge scores.

The script converts family-restricted scores to per-text posteriors, estimates the judge's
per-class prior over all 27,972 texts, and applies a prior-corrected argmax. This experimental pass
uses the unlabeled test pool and was not part of the final submitted system.

  python scripts/experimental_transductive_rerank.py --scores work/family_judge_scores.json \
      --mode sld --beta 1.0 --temp 25 \
      --out work/transductive_predictions.json
"""
import argparse, json, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DEFS = ROOT / "test" / "test_genre_definitions.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--mode", choices=["none", "uniform", "sld"], default="sld")
    ap.add_argument("--beta", type=float, default=1.0, help="damping on the base prior (0..1)")
    ap.add_argument("--temp", type=float, default=25.0, help="softmax temperature over 0-100 scores")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    s2b = {d["specific_genre"]: d["broad_genre"] for d in json.loads(DEFS.read_text())}
    items = json.loads(a.scores.read_text())

    # per-text posteriors over that text's candidate specifics
    posts = []
    for it in items:
        g = it["genres"]; sc = np.array(it["scores"], dtype=float)
        p = np.exp((sc - sc.max()) / a.temp); p /= p.sum()
        posts.append((it["id"], g, p))

    # base prior conditional on candidacy: mean posterior mass a genre gets where it competes
    mass = Counter(); cand = Counter()
    for _, g, p in posts:
        for gi, pi in zip(g, p):
            mass[gi] += pi; cand[gi] += 1
    pi_base = {k: mass[k] / cand[k] for k in mass}          # judge's pick-rate when offered

    if a.mode == "sld":
        # Saerens-Latinne-Decaestecker EM to estimate the true test marginal, then correct.
        pi = dict(pi_base)
        for _ in range(a.iters):
            new_mass = Counter(); new_cand = Counter()
            for _, g, p in posts:
                adj = np.array([p[i] * pi[g[i]] / pi_base[g[i]] for i in range(len(g))])
                adj = adj / adj.sum() if adj.sum() > 0 else p
                for gi, pv in zip(g, adj):
                    new_mass[gi] += pv; new_cand[gi] += 1
            pi = {k: new_mass[k] / cand[k] for k in mass}   # normalise by candidacy, not by count
        corr = {k: (pi[k] / pi_base[k]) ** a.beta for k in mass}
    elif a.mode == "uniform":
        corr = {k: (1.0 / pi_base[k]) ** a.beta for k in mass}
    else:
        corr = {k: 1.0 for k in mass}

    preds = []
    for rid, g, p in posts:
        adj = np.array([p[i] * corr.get(g[i], 1.0) for i in range(len(g))])
        spec = g[int(adj.argmax())]
        preds.append({"id": rid, "broad_genre": s2b[spec], "specific_genre": spec})

    a.out.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"[rerank] mode={a.mode} beta={a.beta} temp={a.temp} -> {len(preds)} preds -> {a.out}")
    print("[rerank] n_distinct_specific:", len({p['specific_genre'] for p in preds}))
    print("[rerank] specific top10:", Counter(p['specific_genre'] for p in preds).most_common(10))


if __name__ == "__main__":
    main()
