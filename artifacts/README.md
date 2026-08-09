# Cached model outputs

These files let `reproduce.sh` rebuild the submitted predictions without a GPU or API access.

| File | Produced by | Notes |
| --- | --- | --- |
| `broad_gate.json` | an earlier run of `src/judge.py` over a wider candidate set | **Not regenerable from this repository.** It supplies the broad family that `src/judge.py` then restricts its specific-genre scoring to, and the fallback specific label when a family filters empty. It is a prediction file, not released data: its `specific_genre` differs from `base_predictions.json` on 5,258 of 27,972 rows. Anyone rerunning the pipeline from scratch inherits this gate rather than reproducing it. |
| `base_predictions.json` | `src/judge.py` | the base system, 0.7139 |
| `cot_predictions.json` | `COT=1 src/judge.py` | chain-of-thought pass used by the attractor drain |
| `verify_gpt.json` | `PROVIDER=gpt src/verify.py` | full-taxonomy re-prediction, all 27,972 |
| `verify_gemini.json` | `PROVIDER=gemini src/verify.py` | the 8,041 GPT-vs-base disagreements; 33 entries are `null` where the model never returned a parsable label |
