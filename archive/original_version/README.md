# Archive — original version of the study

Superseded material, kept for provenance. Nothing here is used by the corrected pipeline
(except that the test evaluation re-evaluates the two original checkpoints, which
remain in `results/models/`).

| Item | Content |
|---|---|
| `paper/data/` | raw records behind its Tables III/IV (`summary.json`, `shapes.json`, traces) |
| `paper/figures/` | its figures |
| `paper/scripts/` | `run_paper_experiments.py`, `run_shape_generalisation.py`, `make_figures.py` (paths updated to run from here) |
| `results/metrics_and_logs/`, `results/plots/`, `results/videos/` | outputs of the original scripts |
| `results/benchmark_summary_root.json` | a benchmark summary that was stored in the repository root |

Three plots of the original version (`fig1_table_iv_lap_times.png`, `fig2_multi_track_precision.png`,
`fig3_interpretability_comparison.png`) and their script `our_model_code/generate_paper_comparison_figures.py`
were removed: they drew hard-coded numbers, not measured results. They remain in the git history.

The audit of this version is in `docs/01_PHASE1_AUDIT.md`: the original numbers are genuine
outputs of the stored checkpoints, but the "Bead-PPO" policy was trained by expert regression and
DAgger (no PPO updates) on its own evaluation circuits. The whole folder can be deleted without
affecting the revised work; it is also in the git history (commit b85c258).
