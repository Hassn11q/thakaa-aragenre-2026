# Cached model outputs

These files let `reproduce.sh` rebuild the submitted predictions without a GPU or API access.

| File | Produced by | Notes |
| --- | --- | --- |
| `broad_gate.json` | submission 866538, an earlier retrieval-gated system of ours that scored 0.6812 | **Not regenerable from this repository.** It supplies the broad family that `src/judge.py` then restricts its specific-genre scoring to, and the fallback specific label when a family filters empty. It is a prediction file, not released data: its `specific_genre` differs from `base_predictions.json` on 5,258 of 27,972 rows. Anyone rerunning the pipeline from scratch inherits this gate rather than reproducing it. |
| `base_predictions.json` | `src/judge.py` **plus post-processing not fully contained in this repository** | The base system, 0.7139. Re-running `src/judge.py` over the same candidate sets reproduces 93.5% of its specific labels and 97.1% of its broad labels, not 100%: this file also carries an Interactive-recall reassignment step that was applied interactively during the competition and was not preserved as a script. Treat it as a cached input, not as something the repository regenerates. |
| `cot_predictions.json` | `COT=1 src/judge.py` | chain-of-thought pass used by the attractor drain |
| `verify_gpt.json` | `PROVIDER=gpt src/verify.py` | full-taxonomy re-prediction, all 27,972 |
| `verify_gemini.json` | `PROVIDER=gemini src/verify.py` | the 8,041 GPT-vs-base disagreements; 33 entries are `null` where the model never returned a parsable label |
| `trap_pool.json` | `src/pools.py` | the 2,635-instance verification pool used by the three-arm topic-trap experiment; 1,403 of them are `*_book_description` |
| `trap_arm_a.json` | `src/verify.py`, broad definitions only | arm A of the trap experiment: no type-versus-topic instruction |
| `trap_arm_b.json` | `src/ablate_trap.py` | arm B: broad definitions plus the explicit type-versus-topic instruction. Arm C is `verify_gpt.json`. `src/trap_table.py` rebuilds the 394/233/126 table from these three files without any API access |
