# Cached model outputs

These files let `reproduce.sh` rebuild the submitted predictions without a GPU or API access.

| File | Produced by | Notes |
| --- | --- | --- |
| `broad_gate.json` | submission 866538, an earlier retrieval-gated system of ours that scored 0.6812 | **Not regenerable from this repository.** It supplies the broad family that `src/judge.py` then restricts its specific-genre scoring to, and the fallback specific label when a family filters empty. It is a prediction file, not released data: its `specific_genre` differs from `base_predictions.json` on 5,258 of 27,972 rows. Anyone rerunning the pipeline from scratch inherits this gate rather than reproducing it. |
| `base_predictions.json` | `src/judge.py` **plus a reassignment step not wired into `reproduce.sh`** | The base system, 0.7139. Re-running `src/judge.py` over the same candidate sets reproduces 93.5% of its specific labels and 97.1% of its broad labels, not 100%: this file also carries an Interactive-recall reassignment produced by `src/interactive_judge.py`, an automatic judge pass that is released here but is not part of the reproduction path. Treat it as a cached input, not as something `reproduce.sh` regenerates. |
| `cot_predictions.json` | `COT=1 src/judge.py` | chain-of-thought pass used by the attractor drain |
| `verify_gpt.json` | `PROVIDER=gpt src/verify.py` | full-taxonomy re-prediction, all 27,972 |
| `verify_gemini.json` | `PROVIDER=gemini src/verify.py` | the 8,041 GPT-vs-base disagreements; 33 entries are `null` where the model never returned a parsable label |
| `trap_pool.json` | a broad-suspect pool selected during the evaluation phase | The 2,635-instance verification pool used by the three-arm topic-trap experiment; 1,403 of them are `*_book_description`. **No producer ships here** — `src/pools.py` writes only `all_ids.json` and `gpt_disagree_ids.json`. |
| `trap_arm_c.json` | `src/verify.py`, full taxonomy, pool run | arm C of the trap experiment, matched to arms A and B on the 2,635-instance pool |
| `trap_arm_a.json` | a broad-only verifier prompt not included in this release | Arm A of the trap experiment: no type-versus-topic instruction, broad labels only. **No producer ships here** — `src/verify.py` always sends the full taxonomy and always returns a specific genre, so it cannot generate this file. |
| `trap_arm_b.json` | `src/ablate_trap.py` | arm B: broad definitions plus the explicit type-versus-topic instruction. Arm C is `trap_arm_c.json`, the same-pool full-taxonomy run; `src/trap_table.py` rebuilds the 394/233/119 table from these files without any API access. `verify_gpt.json` is the separate all-instance run the deployed pipeline uses, and gives 126 on this subset. Arm B was run after the evaluation phase closed, so that comparison is post hoc. |
| `gpt_broad.json`, `gemini_broad.json`, `sonnet_broad.json` | the broad-only verifier prompt, one file per model | Each model's broad-genre verdict over the 2,635-instance pool under the 6-definition prompt, released so the broad-only claims can be checked: all three call 157 of the 1,403 book descriptions Religious unanimously, and GPT and Gemini jointly call 134 of the 548 book descriptions that Gemini also covers under the full taxonomy. `gpt_broad.json` is the same run as `trap_arm_a.json`. **No producer ships here** — `src/verify.py` always sends the full taxonomy. |
| `judge_rerun_scores.json` | `src/judge.py` | a full 27,972-row re-run of the judge over the same candidate sets, released so the 93.5% specific / 97.1% broad figure above can be checked without a GPU |
| `cot_scores.json` | `COT=1 src/judge.py` | the per-candidate scores behind `cot_predictions.json` |
| `early_base.json` | `src/judge.py` | an earlier base rebuild, kept because its score file survives |
| `early_base_scores.json` | `src/judge.py` | the per-candidate scores behind `early_base.json` |
| `final_leaderboard.tsv` | the organisers' final standings | all 18 teams with their seven official metrics; the source for the paper's rank and for the rank-2 comparison in Table 2 |

## Checking the centring constant

The submitted base run's invocation was not logged. These two pairs are the cached predictions
whose score files survive, and both are reproduced exactly at `ALPHA=0.75` and at no other value
we tried:

```sh
python3 src/judge_rerun_check.py --scores artifacts/cot_scores.json \
    --base artifacts/cot_predictions.json --alpha 0.75      # 27972/27972
python3 src/judge_rerun_check.py --scores artifacts/early_base_scores.json \
    --base artifacts/early_base.json --alpha 0.75           # 27972/27972
```
