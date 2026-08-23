# Released verbatim as the script that produced the cached candidate lists the submitted
# base consumed (test_topk_0of2.json / test_topk_1of2.json, written 28 July 2026 09:24 and
# 09:25). It expects the working-repo layout (test/), not this release's data/ directory;
# src/retrieve.py is the cleaned release version of the same configuration. Kept unedited so
# the 4-bit NF4, double-quantisation, fp16-compute and max_seq_length=512 settings reported
# in the paper can be checked against the code that actually ran.
"""Phase A of the shipped genre-general system on the hidden TEST set.

Qwen3-Embedding-8B encodes each Arabic test text and each candidate's SPECIFIC definition
only (spec_def framing). Cosine top-k specific genres per text are saved for the setwise
gemma judge (Phase B). Fully genre-general: no genre NAME enters any text, so it transfers
to the unseen test genres.

Weights are the true Qwen3-Embedding-8B loaded in 4-bit NF4 with fp16 compute (all three
local vLLM installs are ABI/CUDA-broken, and fp16 weights do not fit the ~15 GB free per GPU
left by another user's training job). 4-bit is retrieval-equivalent here: only the cosine
top-k ranking is used, and NF4 preserves embedding direction.
"""
import json, os, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test" / "test.json"
DEFS = ROOT / "test" / "test_genre_definitions.json"
SHARD = os.environ.get("SHARD", "0/1")            # "i/n": which contiguous slice of rows to process
SI, SN = (int(x) for x in SHARD.split("/"))
OUT = ROOT / "scratch_logs" / f"test_topk_{SI}of{SN}.json"
TOPK = 20  # save a deep candidate list so the judge can pick any k without re-embedding
# Qwen3-Embedding query instruction, applied to the Arabic TEXTS only (from the harness MODELS spec).
QPROMPT = "Instruct: Given an Arabic text, retrieve the genre definition that best describes it.\nQuery: "
CHAR_CAP = 8000  # pre-truncate very long texts; genre signal is in the opening (max_model_len=8192).

def main():
    from transformers import BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
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
