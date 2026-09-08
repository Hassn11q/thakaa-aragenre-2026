# Data provenance

All JSON files are UTF-8 and contain only task IDs, taxonomy labels, or released taxonomy
definitions. Hidden-test text and gold labels are not included.

## Submitted-system files

| Path | Role |
|---|---|
| `cached_outputs/family_gate_predictions.json` | Earlier official submission reused as the broad-family gate |
| `cached_outputs/base_predictions.json` | Five-step base system consumed by the final assembly |
| `cached_outputs/gpt_verifier_predictions.json` | GPT 5.6 Luna full-taxonomy output for all 27,972 IDs |
| `cached_outputs/gemini_verifier_predictions.json` | Gemini 3.6 Flash output on GPT 5.6 Luna/base disagreements |
| `cached_outputs/rationalized_judge_predictions.json` | Judge pass consumed by the attractor drain |
| `cached_outputs/surface_rule_overrides.json` | Text-free cache of the submitted surface-rule decisions |
| `submissions/final_submission.json` | Exact official prediction file associated with the 0.7352 score |
| `submissions/final_submission.zip` | Release ZIP containing the exact submitted JSON as `predictions.json` |

`run_reproduce.sh` reconstructs the final JSON/ZIP from the cached outputs and refuses success
unless every reconstructed prediction matches `submissions/final_submission.json`.

Prompt strings are preserved verbatim, including their capitalization and punctuation. Editing
their wording would no longer document the submitted system.

## Post-evaluation diagnostics

Files under `cached_outputs/diagnostics/` are post-evaluation outputs used only for the prompt-context
analysis. `scripts/rebuild_prompt_diagnostics.py` regenerates the reported three-arm counts,
base-family agreement table, and paired two-verifier comparison. These diagnostics did not
contribute to the official submission.

## Known reproducibility boundaries

- The public bundle reconstructs the submitted output exactly from cached model decisions.
- It does not regenerate the entire base pipeline bit-for-bit.
- Hosted-model reruns may change over time.
- The submitted judge serving precision and model revision hashes were not recorded.
- The three documented regex defects are preserved because correcting them changes the scored
  output.
