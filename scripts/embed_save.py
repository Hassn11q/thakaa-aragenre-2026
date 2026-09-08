"""Embed test texts and candidate definitions with Qwen3-Embedding-8B in fp16.

The saved vectors support retrieval and the documented transductive diagnostics.

Outputs are ``work/emb_qv.npy`` (N x 4096, L2-normalized), ``emb_ids.json``,
``emb_cv.npy`` (74 x 4096), and ``emb_cand.json``. Arrays use float16.
"""
import json, os, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test" / "test.json"
DEFS = ROOT / "test" / "test_genre_definitions.json"
OUT = Path(os.environ.get("OUT_DIR", ROOT / "work"))
OUT.mkdir(parents=True, exist_ok=True)
QPROMPT = "Instruct: Given an Arabic text, retrieve the genre definition that best describes it.\nQuery: "
CHAR_CAP = 8000
MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "Qwen/Qwen3-Embedding-8B")

def main():
    from sentence_transformers import SentenceTransformer
    rows = json.loads(TEST.read_text())
    defrows = json.loads(DEFS.read_text())
    ids = [r["id"] for r in rows]
    texts = [(r.get("text") or "")[:CHAR_CAP] for r in rows]
    cand = [d["specific_genre"] for d in defrows]
    cand_def = [d["specific_genre_definition"] for d in defrows]

    model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device="cuda:0",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 512

    def emb(strs, bs):
        return model.encode(strs, normalize_embeddings=True, convert_to_numpy=True,
                            batch_size=bs, show_progress_bar=True).astype(np.float16)

    cv = emb(cand_def, 16)
    qv = emb([QPROMPT + t for t in texts], 24)
    np.save(OUT / "emb_qv.npy", qv)
    np.save(OUT / "emb_cv.npy", cv)
    (OUT / "emb_ids.json").write_text(json.dumps(ids))
    (OUT / "emb_cand.json").write_text(json.dumps(cand))
    print(f"[embed_save] qv {qv.shape} cv {cv.shape} saved to {OUT}")

if __name__ == "__main__":
    main()
