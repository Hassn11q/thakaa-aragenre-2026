# Thakaa at AraGenre 2026

Code and cached model outputs for the first-place system in the
[AraGenre 2026](https://www.codabench.org/competitions/16356) shared task on hierarchical
definition-guided Arabic genre classification (ArabicNLP 2026).

**Paper:** [`paper/aragenre_thaka.pdf`](paper/aragenre_thaka.pdf) · **Submission:** 872515 ·
**Rank:** 1 of 18 teams

Every text is assigned one broad genre out of 6 and one specific genre out of 74. The 74
specific genres never occur in the released training data and are given only as English
definitions, so no label can be learned from examples. The system does no task-specific
training.

- [Result](#result)
- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Method](#method)
- [Reproducing the paper](#reproducing-the-paper)
- [Running the pipeline from scratch](#running-the-pipeline-from-scratch)
- [What this artifact cannot regenerate](#what-this-artifact-cannot-regenerate)
- [Repository layout](#repository-layout)
- [Data, licensing and intended use](#data-licensing-and-intended-use)
- [Citing](#citing)

## Result

Official evaluation, hidden test set, 27,972 instances, no missing predictions.

| Metric | Ours | Rank 2 |
| --- | --- | --- |
| **Hierarchical Macro F1** (ranking metric) | **0.7352** | 0.7169 |
| Specific Macro F1 | **0.5725** | 0.5339 |
| Specific Weighted F1 | **0.6125** | 0.5813 |
| Specific Accuracy | **0.6189** | 0.5987 |
| Broad Macro F1 | 0.8979 | **0.9000** |
| Broad Weighted F1 | **0.9142** | 0.9123 |
| Broad Accuracy | **0.9138** | 0.9113 |

The full 18-team standings are in [`artifacts/final_leaderboard.tsv`](artifacts/final_leaderboard.tsv).

## Quickstart

Rebuilds the exact file that was scored. No GPU, no API keys, no network, no third-party
packages. Runs in about a second.

```sh
cp /path/to/test.json data/test.json     # the organisers' to distribute; see data/README.md
./reproduce.sh
```

```
identical predictions: 27972/27972
```

## Requirements

| Path | Needs |
| --- | --- |
| `./reproduce.sh` and every checker | Python 3.9+, standard library only |
| `src/retrieve.py` | CUDA GPU, `torch`, `transformers`, `sentence-transformers`, `bitsandbytes` |
| `src/judge.py`, `src/interactive_judge.py` | an OpenAI-compatible endpoint serving the judge model |
| `src/verify.py`, `src/ablate_trap.py` | `OPENAI_API_KEY` and `GEMINI_API_KEY`; about US$50 for the full test set |

`pip install -r requirements.txt` covers the second group onward. Nothing reads a `.env` file;
export variables in your shell.

## Method

```
Arabic text
  ├─ retrieve.py    Qwen3-Embedding-8B ranks the 74 English definitions → candidate set
  ├─ judge.py       Gemma judge scores candidates setwise inside one broad family
  ├─ rules.py       surface overrides (mushaf orthography, isnad openings, emoji posts)
  ├─ interactive_judge.py
  │                 binary recall pass over short or marker-bearing texts; moved 1,441
  │                 predictions to an Interactive subtype
  │                 → 0.7139
  ├─ verify.py      GPT-5.6 and Gemini-3.6 re-predict with all 74 definitions in context;
  │                 a correction lands only where both models agree against the base
  ├─ assemble.py    attractor drain: over-predicted generic classes are redistributed to
  │                 within-family siblings by a chain-of-thought re-judge, then the
  │                 surface rules are re-applied as a guard
  └─ submission.py  forces broad = parent(specific), validates every id, writes the zip
                    → 0.7352
```

Three findings the paper reports in full:

**Derive the broad label, never predict it.** Our submitted configuration whose broad label came
from a direct six-way judge scored 0.6206, against 0.7139 for the pipeline that derives it from
the specific prediction. The two submissions differ in more than that one choice, so this is a
comparison between configurations, not an isolated ablation.

**Show the model every label, not just the top of the hierarchy.** Given only the 6 broad
definitions, all three frontier models we tried filed a book *about* religion under Religious.
A three-arm ablation over 1,403 book descriptions separates the causes: an explicit
type-versus-topic instruction removes 41% of those errors (394 → 233), and supplying all 74
definitions removes a further 49% (233 → 119). Neither factor alone explains the effect.

**Agreement between two models beats either alone.** A single model disagreed with the base
pipeline on 28.7% of specific labels (8,041 of 27,972), far too noisy to apply. Requiring both models to name the
same label reduced that to corrections worth taking.

Approaches that lost — uniform-prior and optimal-transport calibration, plain self-consistency,
an unrestricted chain-of-thought pass, cross-lingual rerankers — are reported with their scores
in the paper's appendix and in `artifacts/final_leaderboard.tsv`.

## Reproducing the paper

Each checker reads cached outputs and prints the figure the paper quotes.

```sh
python3 src/trap_table.py           # topic-trap ablation      394 -> 233 -> 119
python3 src/dialect_markers.py      # song-lyric dialect ceiling      87-99%
python3 src/orthographic_signals.py # diacritic and emoji coverage    77% / 47%
python3 src/judge_rerun_check.py    # judge reproduction       93.5% / 97.1%
```

`trap_table.py` and `judge_rerun_check.py` need no data at all; the other two read
`data/test.json`.

## Running the pipeline from scratch

Everything writes into `work/`, so the cached files in `artifacts/` stay intact and
`./reproduce.sh` keeps reproducing the scored submission. The final command assembles your own
outputs, so you can diff them against ours.

```sh
export OPENAI_API_KEY=...   GEMINI_API_KEY=...
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

## What this artifact cannot regenerate

Stated plainly, because the reproduction above is exact and these gaps are not.

**The broad gate.** `src/judge.py` restricts its scoring to a broad family taken from
`artifacts/broad_gate.json`. That file is submission 866538, an earlier system of ours that
scored 0.6812; it is shipped and reused, not regenerated. A from-scratch run inherits it.

**The base predictions.** `artifacts/base_predictions.json` is a `src/judge.py` run plus the
`src/interactive_judge.py` reassignment applied during the evaluation phase. Re-running the
judge alone recovers 93.5% of its specific labels and 97.1% of its broad labels; run
`python3 src/judge_rerun_check.py` to confirm that yourself.

**The score-centring constant.** The submitted base invocation was not logged. Our judge run
logs report `ALPHA=0.75`, while sibling experiments that select it over `{0, 0.25, 0.5, 0.75, 1}`
recorded other values. Two cached prediction files whose score files survive reproduce at 0.75
and at no other value on that grid, which is why `src/judge.py` defaults to it:

```sh
python3 src/judge_rerun_check.py --scores artifacts/cot_scores.json \
    --base artifacts/cot_predictions.json --alpha 0.75        # 27972/27972
python3 src/judge_rerun_check.py --scores artifacts/early_base_scores.json \
    --base artifacts/early_base.json --alpha 0.75             # 27972/27972
```

We report 0.75 as a reconstruction rather than a directly verified invocation setting.

**Preserved rule defects.** Three regexes in `src/rules.py` do not match their stated intent: the
Qur'anic class also admits U+0671, the opinion class lets a bare variation selector match, and one
character class contains a literal `|`. They ran as written when the submission was produced, so
they are kept verbatim and documented in place rather than corrected; fixing all three changes 22
of the 27,972 final predictions.

Two inputs to the topic-trap ablation, `artifacts/trap_pool.json` and
`artifacts/trap_arm_a.json`, also have no producer in this repository. See
[`artifacts/README.md`](artifacts/README.md) for the provenance of every cached file.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | one stage per file, plus the four checkers above |
| `artifacts/` | cached model outputs, the final leaderboard, and a README describing each |
| `submissions/` | the file that was scored, JSON and zip |
| `data/` | the released genre definitions; `test.json` is not redistributed |
| `paper/` | system description paper, LaTeX source and PDF |
| `work/` | scratch space, gitignored |

## Data, licensing and intended use

The hidden evaluation set belongs to the shared-task organisers and is **not** redistributed
here; `data/test.json` is gitignored. Only the released genre definitions ship. Every file in
`artifacts/` contains ids and labels, never source text.

The code is MIT (see [LICENSE](LICENSE)). The AraGenre data is governed by the shared task's own
terms. The models the pipeline calls — Qwen3-Embedding-8B, a Gemma-family judge, GPT-5.6 Luna,
Gemini-3.6 Flash — carry their own licences and terms.

This system was built for one benchmark. It classifies text type, not quality, correctness or
orthodoxy, and the taxonomy covers religiously and culturally sensitive categories where a
misclassification carries no judgement of the text. Five of the 74 classes separate song lyrics
by regional dialect, a distinction that is largely phonetic and mostly absent from writing:
87-99% of the texts we assign to them carry no marker from their dialect's own cue list. Treat
fine-grained output as triage rather than a verdict.

## Citing

```bibtex
@inproceedings{alqaeri-etal-2026-thakaa,
  title     = {Thakaa at AraGenre 2026: Definition-Guided LLM Judging with Full-Taxonomy
               Agreement-Gated Correction for Hierarchical Arabic Genre Classification},
  author    = {Alqaeri, Hassan and Alamr, Meshal and Aldahlawi, Abdullah},
  booktitle = {Proceedings of the 4th Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  year      = {2026}
}
```

Please also cite the shared task overview paper, El-Haj et al. (2026).
