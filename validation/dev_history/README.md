# Development History (archived, not part of the reported pipeline)

This folder preserves two early exploratory scripts that were **not** used to
produce any table, figure, or number reported in the paper. They are kept
here — instead of deleted — purely for transparency and provenance, so the
development trail behind the final benchmark (`validation/01a_run_loso_benchmark.py`
through `validation/06b_generate_figure4_confusion_matrices.py`) is auditable.

| File | What it was | Status |
|---|---|---|
| `diagnose_minirocket_exploration.py` | An early diagnostic exploring a MiniROCKET-based classifier as an alternative to the hand-crafted-feature pipeline. | **Not runnable** — depends on a `minirocket` module that was later removed from the project. This direction was abandoned in favour of the hand-crafted-feature approach reported in the paper. |
| `prototype_multi_classifier_runner.py` | An early multi-classifier LOSO comparison script (RandomForest / ExtraTrees / SVM). | Superseded by `validation/01a_run_loso_benchmark.py`, which uses the final classifier set (XGBoost, LDA, LinearSVC, RandomForest) and protocol reported in the paper. |

**If you are trying to reproduce the paper's results, you do not need
anything in this folder.** Start from the main [README](../../README.md)
instead.
