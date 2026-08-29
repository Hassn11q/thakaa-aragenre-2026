# Explored, never scored: no submitted configuration used this pass. Shipped because the
# paper's Negative Results appendix records it as a built-but-unscored path. It needs the
# cached test embeddings, which are not redistributed, so it does not run from this release.
"""Transductive TestNUC label smoothing over the kNN graph of the 27,972 test embeddings.

Same-genre texts cluster in Qwen3 space, so a text's nearest neighbours vote on its label. This
corrects outliers and — because broad = parent(specific) — repairs FAMILY (broad) assignment, our
weak axis. Self is always a voter (weight 1), so confident texts are not overridden by weak crowds.

  python scripts/label_propagation.py --pred submissions/test_v3_hier.json \
      --k 40 --sim 0.55 --margin 1.15 --out scratch_logs/test_v3_prop.json
"""
import argparse, json, numpy as np, torch
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch_logs"
DEFS = ROOT / "test" / "test_genre_definitions.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--sim", type=float, default=0.55, help="min cosine to count a neighbour")
    ap.add_argument("--margin", type=float, default=1.15,
                    help="override self only if best-other weight > margin * self-label weight")
    ap.add_argument("--level", choices=["specific", "broad", "family"], default="family")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    s2b = {d["specific_genre"]: d["broad_genre"] for d in json.loads(DEFS.read_text())}
    ids = json.loads((OUT / "emb_ids.json").read_text())
    qv = torch.tensor(np.load(OUT / "emb_qv.npy"), dtype=torch.float16, device="cuda:0")  # N x d, normalized
    cand = json.loads((OUT / "emb_cand.json").read_text())
    cv = torch.tensor(np.load(OUT / "emb_cv.npy"), dtype=torch.float16, device="cuda:0")   # 74 x d
    pred = {p["id"]: p for p in json.loads(a.pred.read_text())}
    spec0 = [pred[i]["specific_genre"] for i in ids]

    if a.level == "family":
        # Propagate the FAMILY (parent). Because build forces broad=parent(specific), when a text's
        # family flips we must also re-pick a valid in-family specific (nearest definition by cosine).
        lab = [s2b[s] for s in spec0]
    else:
        lab = list(spec0)                          # 74-way specific propagation

    # precompute per-family specific indices for in-family re-pick
    fam_specs = {}
    for j, s in enumerate(cand):
        fam_specs.setdefault(s2b[s], []).append(j)

    N = len(ids)
    new_spec = list(spec0)
    changed = 0
    B = 2048
    for s in range(0, N, B):
        q = qv[s:s+B]
        sims = (q @ qv.T).float()
        # cosine of these texts to every definition (for in-family re-pick)
        dsim = (q.float() @ cv.float().T)          # B x 74
        for r in range(q.shape[0]):
            i = s + r
            row = sims[r]; row[i] = -1.0
            vals, idx = torch.topk(row, a.k)
            w = Counter(); w[lab[i]] += 1.0
            for v, j in zip(vals.tolist(), idx.tolist()):
                if v >= a.sim:
                    w[lab[j]] += v
            best, bw = max(w.items(), key=lambda kv: kv[1])
            if best != lab[i] and bw > a.margin * w[lab[i]]:
                changed += 1
                if a.level == "family":
                    fj = fam_specs[best]
                    new_spec[i] = cand[fj[int(dsim[r, fj].argmax())]]   # nearest def in the new family
                else:
                    new_spec[i] = best

    preds = [{"id": rid, "broad_genre": s2b[new_spec[i]], "specific_genre": new_spec[i]}
             for i, rid in enumerate(ids)]
    a.out.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    print(f"[prop] level={a.level} k={a.k} sim={a.sim} margin={a.margin} changed {changed}/{N}")
    print("[prop] broad:", Counter(p["broad_genre"] for p in preds).most_common())


if __name__ == "__main__":
    main()
