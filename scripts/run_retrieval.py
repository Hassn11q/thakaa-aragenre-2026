"""Retrieve candidate genres for the hidden test set.

Qwen3-Embedding-8B encodes each Arabic test text and each candidate's specific definition.
The script saves the top-k genres by cosine similarity for the family-restricted judge. It does
not append genre names to test texts.

Weights are loaded in 4-bit NF4 with fp16 compute. Only the cosine top-k ranking is consumed;
the paper reports the reproducibility limits of this quantized retrieval stage.
"""
import json, os, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test" / "test.json"
DEFS = ROOT / "test" / "test_genre_definitions.json"
SHARD = os.environ.get("SHARD", "0/1")            # "i/n": which contiguous slice of rows to process
SI, SN = (int(x) for x in SHARD.split("/"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "work"))
OUT = OUT_DIR / f"retrieval_topk_{SI}of{SN}.json"
TOPK = 20  # save a deep candidate list so the judge can pick any k without re-embedding
# Qwen3-Embedding query instruction, applied only to the Arabic texts.
QPROMPT = "Instruct: Given an Arabic text, retrieve the genre definition that best describes it.\nQuery: "
CHAR_CAP = 8000  # pre-truncate very long texts; genre signal is in the opening (max_model_len=8192).

def main():
    from transformers import BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allrows = json.loads(TEST.read_text())
    rows = allrows[SI::SN]                          # strided shard: process 1 of every SN rows
    defrows = json.loads(DEFS.read_text())
    cand_spec = [d["specific_genre"] for d in defrows]
    cand_def = [d["specific_genre_definition"] for d in defrows]      # spec_def framing: definition only
    spec2broad = {d["specific_genre"]: d["broad_genre"] for d in defrows}

    texts = [(r.get("text") or "")[:CHAR_CAP] for r in rows]
    q_in = [QPROMPT + t for t in texts]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", trust_remote_code=True, device="cuda:0",
                                model_kwargs={"quantization_config": bnb, "torch_dtype": torch.float16})
    model.max_seq_length = 512  # cap activations under the ~14 GB free; genre signal is early in the text

    def embed(strs):
        return model.encode(strs, normalize_embeddings=True, convert_to_numpy=True,
                            batch_size=10, show_progress_bar=True).astype(np.float32)

    cv = embed(cand_def)                 # candidate definitions (no instruction prompt)
    qv = embed(q_in)                     # test texts (with instruction prompt)
    order = np.argsort(-(qv @ cv.T), axis=1)

    out = []
    for i, r in enumerate(rows):
        top = order[i][:min(TOPK, len(cand_spec))]
        out.append({"id": r["id"],
                    "top_genres": [cand_spec[j] for j in top],
                    "top_defs": [cand_def[j] for j in top]})
    OUT.write_text(json.dumps({"topk": out, "spec2broad": spec2broad}, ensure_ascii=False))
    print(f"[phaseA] {len(out)} rows, top{TOPK} saved -> {OUT}")
    # quick sanity: retrieval-top1 genre distribution
    from collections import Counter
    c = Counter(o["top_genres"][0] for o in out)
    print("[phaseA] top1 spread (most common 8):", c.most_common(8))

if __name__ == "__main__":
    main()
