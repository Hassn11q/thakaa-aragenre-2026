"""Rank the 74 genre definitions against every test text.

Qwen3-Embedding-8B encodes each Arabic text and each candidate's specific definition, and the
cosine top-k specific genres per text are written for the setwise judge to score. No genre
name ever enters the encoded text, so the ranking transfers to unseen genres.

Runs in fp16 on GPU and float32 on CPU; set RETRIEVE_DEVICE to choose, MAX_SEQ_LEN to cap
sequence length, and SHARD ("i/n") to split the work across processes.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# Set RETRIEVE_DEVICE to cuda:1 or cpu when the default GPU is busy or absent.
DEVICE = os.environ.get("RETRIEVE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
TEST = ROOT / "data" / "test.json"
DEFS = ROOT / "data" / "test_genre_definitions.json"
SHARD = os.environ.get("SHARD", "0/1")  # "i/n": which contiguous slice of rows to process
SI, SN = (int(x) for x in SHARD.split("/"))
OUT = ROOT / "work" / f"candidates_{SI}of{SN}.json"
TOPK = 20  # save a deep candidate list so the judge can pick any k without re-embedding
# Qwen3-Embedding query instruction, applied to the Arabic TEXTS only (from the harness MODELS spec).
QPROMPT = (
    "Instruct: Given an Arabic text, retrieve the genre definition that best describes it.\nQuery: "
)
CHAR_CAP = (
    8000  # pre-truncate very long texts; genre signal is in the opening (max_model_len=8192).
)


def main():
    """Rank the genre definitions against each text and write the candidate sets."""
    from sentence_transformers import SentenceTransformer

    allrows = json.loads(TEST.read_text())
    rows = allrows[SI::SN]  # strided shard: process 1 of every SN rows
    defrows = json.loads(DEFS.read_text())
    cand_spec = [d["specific_genre"] for d in defrows]
    cand_def = [
        d["specific_genre_definition"] for d in defrows
    ]  # spec_def framing: definition only
    spec2broad = {d["specific_genre"]: d["broad_genre"] for d in defrows}

    texts = [(r.get("text") or "")[:CHAR_CAP] for r in rows]
    q_in = [QPROMPT + t for t in texts]

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        trust_remote_code=True,
        device=DEVICE,
        model_kwargs={"torch_dtype": DTYPE},
    )
    model.max_seq_length = MAX_SEQ_LEN

    def embed(strs):
        """Encode a list of strings into L2-normalised vectors."""
        return model.encode(
            strs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=10,
            show_progress_bar=True,
        ).astype(np.float32)

    cv = embed(cand_def)  # candidate definitions (no instruction prompt)
    qv = embed(q_in)  # test texts (with instruction prompt)
    order = np.argsort(-(qv @ cv.T), axis=1)

    out = []
    for i, r in enumerate(rows):
        top = order[i][: min(TOPK, len(cand_spec))]
        out.append(
            {
                "id": r["id"],
                "top_genres": [cand_spec[j] for j in top],
                "top_defs": [cand_def[j] for j in top],
            }
        )
    OUT.write_text(json.dumps({"topk": out, "spec2broad": spec2broad}, ensure_ascii=False))
    print(f"[phaseA] {len(out)} rows, top{TOPK} saved -> {OUT}")
    # quick sanity: retrieval-top1 genre distribution
    from collections import Counter

    c = Counter(o["top_genres"][0] for o in out)
    print("[phaseA] top1 spread (most common 8):", c.most_common(8))


if __name__ == "__main__":
    main()
