# Thakaa at AraGenre 2026: first-place system

This repository contains the exact submitted predictions, text-free cached model outputs,
prompts, and scripts for Thakaa's AraGenre 2026 system.

## Official result

| Hier. Macro F1 | Spec. Macro | Spec. Wtd | Spec. Acc | Broad Macro | Broad Wtd | Broad Acc |
|---|---|---|---|---|---|---|
| **0.7352** (rank 1 of 18) | 0.5725 | 0.6125 | 0.6189 | 0.8979 | 0.9142 | 0.9138 |

The system uses no task-specific weight updates. It is zero-shot with respect to the 74
hidden-test specific genres, but it includes task-specific prompts, hand-written cues,
manual output inspection, transductive operations, and leaderboard-guided component selection.
The release documents these qualifications in detail.

## Exact offline reconstruction

No API key, GPU, or hidden-test text is needed:

```bash
bash run_reproduce.sh
```

The script reconstructs the final submission from cached outputs, validates every label and
the broad/specific hierarchy, and checks all 27,972 predictions against the submitted file.
Expected final output:

```text
identical predictions: 27972/27972
OK: reproduction matches the submitted system.
```

To verify the downloaded bundle before running it:

```bash
sha256sum -c SHA256SUMS
```

The official test text is not redistributed. The final surface-rule decisions are supplied as
the text-free `cached_outputs/surface_rule_overrides.json`, which contains only IDs and labels.

## Pipeline

```text
Arabic text
  -> Qwen3-Embedding-8B retrieval over the 74 English definitions
  -> google/gemma-4-31B-it family-restricted setwise judge
  -> GPT-5.6 Luna + Gemini 3.6 Flash agreement-gated correction
  -> within-family attractor drain
  -> preserved submitted surface-rule decisions
  -> broad = parent(specific), validation, and ZIP packaging
```

## Optional model reruns

Model reruns are not bit-reproducible because hosted models can change and the submitted local
judge's serving precision and checkpoint revisions were not recorded. They also require the
official `test.json`, which must be obtained from the shared-task organizers and placed at
`test/test.json`.

Copy `.env.example` to `.env`, fill in the required values, and export them:

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
set -a
. ./.env
set +a
```

Then run the desired stages:

```bash
# Retrieval (GPU). Multiple shards may be run independently.
SHARD=0/1 OUT_DIR=work python3 scripts/run_retrieval.py

# Family-restricted judge served through an OpenAI-compatible local endpoint.
KTOP=6 WORKERS=64 ALPHA=0.75 OUT=work/family_judge_predictions.json \
  python3 scripts/run_family_judge.py

# Rationalized pass used by the attractor drain.
COT=1 KTOP=6 WORKERS=64 ALPHA=0.75 OUT=work/rationalized_judge_predictions.json \
  python3 scripts/run_family_judge.py

# Full-taxonomy hosted verifiers.
python3 scripts/make_pools.py
PROVIDER=gpt POOL=work/all_ids.json OUT=work/gpt_verifier_predictions.json \
  python3 scripts/run_full_taxonomy_verifiers.py
python3 scripts/make_pools.py --gpt work/gpt_verifier_predictions.json
PROVIDER=gemini POOL=work/gpt_disagree_ids.json OUT=work/gemini_verifier_predictions.json \
  python3 scripts/run_full_taxonomy_verifiers.py
```

The exact submitted assembly always uses the files in `cached_outputs/`; rerun outputs go to the
ignored `work/` directory and do not overwrite the evidence bundle.

The reported post-evaluation prompt diagnostics can also be rebuilt without API access:

```bash
python3 scripts/rebuild_prompt_diagnostics.py
```

## Repository layout

```text
cached_outputs/  text-free model decisions used by the submitted assembly
scripts/      prompts, model-stage scripts, assembly, validation, and audits
submissions/  the exact official JSON and ZIP submitted for scoring
test/         the released taxonomy definitions; hidden-test text is excluded
```

The three known regex defects from the submitted surface-rule implementation are intentionally
preserved in `scripts/apply_rules.py`. Fixing them would no longer reconstruct the official scored
system.

## Citation

```bibtex
@inproceedings{alqaeri-etal-2026-thakaa,
  title = {Thakaa at AraGenre 2026: Definition-Guided LLM Judging with Full-Taxonomy
           Agreement-Gated Correction for Hierarchical Arabic Genre Classification},
  author = {Alqaeri, Hassan and Alamr, Meshal and Aldahlawi, Abdullah},
  booktitle = {Proceedings of the 4th Arabic Natural Language Processing Conference
               (ArabicNLP 2026)},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  year = {2026}
}
```

Please also cite the shared-task overview:

```bibtex
@inproceedings{elhaj-etal-2026-aragenre,
  title = {AraGenre 2026: A Hierarchical Definition-Guided Arabic Genre Classification Shared Task},
  author = {El-Haj, Mo and Ezzini, Saad and Abudalfa, Shadi and Jarrar, Mustafa and
            Chi, Nguyen Minh and Quan, Nguyen Minh},
  booktitle = {Proceedings of the 4th Arabic Natural Language Processing Conference
               (ArabicNLP 2026)},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  year = {2026}
}
```
