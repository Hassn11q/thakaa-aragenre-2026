# Thakaa at AraGenre 2026

First-place system for the [AraGenre 2026](https://www.codabench.org/competitions/16356) shared
task on hierarchical Arabic genre classification (ArabicNLP 2026).

Each Arabic text gets one of 6 broad genres and one of 74 specific genres. The 74 specific
genres never appear in the training data and are described only by English definitions, so
nothing can be learned from labels. This system does no task training.

## Result

Hidden test set, 27,972 instances:

| Metric | Score |
| --- | --- |
| Hierarchical Macro F1 (official ranking metric) | **0.7352** |
| Specific Macro F1 | 0.5725 |
| Specific Weighted F1 | 0.6125 |
| Specific Accuracy | 0.6189 |
| Broad Macro F1 | 0.8979 |
| Broad Weighted F1 | 0.9142 |
| Broad Accuracy | 0.9138 |

## How it works

```
Arabic text
  ├─ retrieve.py    Qwen3-Embedding-8B ranks the 74 English definitions → candidate set
  ├─ judge.py       Gemma judge scores candidates setwise, restricted to one broad family
  ├─ rules.py       surface overrides (mushaf orthography, isnad openings, emoji posts)
  │                 → 0.7139
  ├─ verify.py      GPT-5.6 and Gemini-3.6 re-predict with all 74 definitions in context;
  │                 a correction lands only when both models agree against the base
  ├─ assemble.py    attractor drain: over-predicted generic classes are redistributed to
  │                 within-family siblings by a chain-of-thought re-judge
  └─ submission.py  forces broad = parent(specific), validates every id, writes the zip
                    → 0.7352
```

Broad genre is never predicted directly. It is derived from the specific prediction, which
beat a direct six-way classifier by a wide margin (0.7139 vs 0.6206).

## Reproduce the submitted run

The model outputs are cached in `artifacts/`, so this needs no GPU, no API keys, and no
third-party packages. Python 3.9+ and the standard library are enough.

```sh
# no dependencies needed for this step
cp /path/to/test.json data/test.json     # not redistributed here
./reproduce.sh
```

The script rebuilds the submission and checks it against the file that scored 0.7352.
Expected output: `identical predictions: 27972/27972`.

## Run the pipeline from scratch

Needs a CUDA GPU for the retriever, a vLLM endpoint for the judge, and API keys for the two
verifiers. Roughly $50 of API calls for the full test set.

Everything below writes into `work/`, so the cached files in `artifacts/` stay intact and
`./reproduce.sh` keeps reproducing the scored submission. The last two commands assemble your
own outputs instead of the cached ones, so you can compare the two.

```sh
export OPENAI_API_KEY=...   GEMINI_API_KEY=...   # read from the environment, not from .env
export JUDGE_URL=...        JUDGE_MODEL=...      # your own OpenAI-compatible endpoint

python src/retrieve.py                   # candidate sets -> work/candidates_*.json
python src/judge.py                      # base predictions -> work/judge_predictions.json
COT=1 OUT=work/cot_predictions.json python src/judge.py

python src/pools.py                      # -> artifacts/all_ids.json
PROVIDER=gpt POOL=artifacts/all_ids.json OUT=work/verify_gpt.json python src/verify.py
python src/pools.py --gpt work/verify_gpt.json
PROVIDER=gemini POOL=artifacts/gpt_disagree_ids.json OUT=work/verify_gemini.json python src/verify.py

python src/assemble.py --base work/judge_predictions.json --gpt work/verify_gpt.json \
  --gemini work/verify_gemini.json --cot work/cot_predictions.json --out work/my_predictions.json
```

Two caveats. `src/judge.py` reads `artifacts/broad_gate.json`, which this repository ships but
cannot regenerate (see `artifacts/README.md`), so a from-scratch run inherits that gate. And
`artifacts/base_predictions.json` additionally carries the `src/interactive_judge.py` pass, which
is not in the sequence above; re-running the judge alone reproduces about 93.5% of its specific
labels.

## Verify the paper's numbers

Each of these reads cached outputs only. No GPU, no API keys, no third-party packages.

```sh
python3 src/trap_table.py           # topic-trap ablation, 394 -> 233 -> 119
python3 src/dialect_markers.py      # song-lyric dialect ceiling, 87-99% carry no marker
python3 src/orthographic_signals.py # diacritic and emoji coverage behind the no-preprocessing choice
python3 src/judge_rerun_check.py    # 93.5% specific / 97.1% broad judge reproduction
```

`trap_table.py` needs no data at all; the other three read `data/test.json`.

`src/ablate_trap.py` and `src/enrich_definitions.py` need API access and produced cached inputs;
`src/enrich_definitions.py` produced none of the shipped artifacts and is included only for
completeness. `src/discriminators.py` is **not** an offline generator: it is a static table of 74
hand-written Arabic cues that `src/judge.py` appends to every definition at run time (`ENRICH=1`
by default), so it is part of the submitted system.
`src/interactive_judge.py` is the reassignment pass that `artifacts/base_predictions.json`
carries; it needs the judge endpoint.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | pipeline, one stage per file, plus the checkers under "Verify the paper's numbers" |
| `artifacts/` | cached model outputs, enough to rebuild the submission offline |
| `submissions/` | the file that was scored |
| `data/` | genre definitions; `test.json` is the organisers' to distribute |
| `paper/` | system description paper, LaTeX and PDF |
| `work/` | scratch space for intermediate files |

## What we learned

**Derive the broad label, do not predict it.** Forcing `broad = parent(specific)` beat a
direct six-way broad classifier by 0.09 Hierarchical Macro F1.

**Show the model every label, not just the top of the hierarchy.** Given only the 6 broad
definitions, all three frontier models we tried filed a book *about* religion under Religious.
An ablation over 1,403 book descriptions separates the two causes: an explicit
type-versus-topic instruction removes 41% of those errors, and supplying all 74 definitions
removes 46% of what remains. Neither alone accounts for the effect.

**Agreement between two models beats either model alone.** A single model disagreed with the
base pipeline on about 29% of specific labels, far too noisy to apply. Requiring both models
to name the same label cut that to a set of corrections worth taking.

**Things that lost.** Uniform-prior and optimal-transport calibration, plain self-consistency,
cross-lingual rerankers, and hand-written regex rules all scored below the 0.7139 base. The
numbers are in the paper's appendix.

## Citing

```bibtex
@inproceedings{thakaa-aragenre-2026,
  title     = {Thakaa at AraGenre 2026: Definition-Guided LLM Judging with Full-Context
               Multi-Model Consensus for Hierarchical Arabic Genre Classification},
  author    = {Alqaeri, Hassan and Alamr, Meshal},
  booktitle = {Proceedings of ArabicNLP 2026},
  year      = {2026}
}
```

Please also cite the shared task overview paper (El-Haj et al., 2026).

## License

MIT, see [LICENSE](LICENSE).
