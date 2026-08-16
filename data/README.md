# Data

| File | Status |
| --- | --- |
| `test_genre_definitions.json` | included: the 74 specific genres and 6 broad genres with their English definitions, exactly as released by the organisers |
| `test.json` | **not included** |

`test.json` is the hidden evaluation set: 27,972 Arabic texts that the AraGenre 2026
organisers own and distribute. We do not redistribute it. Obtain it from the shared task and
place it at `data/test.json`, after which `./reproduce.sh` rebuilds the submitted predictions.

Each row of `test.json` is `{"id": ..., "text": ...}`. The pipeline reads `id` and `text` and
nothing else, so any file with those two fields works for a dry run.

Every file under `artifacts/` is a model output over this input, not released data. See
[`../artifacts/README.md`](../artifacts/README.md) for what each one is and which of them the
released code cannot regenerate.
