# Direct-Actuation Bead Tracing versus RL-Tuned Pure Pursuit

Image → ordered reference path → 2-D point-mass ("bead") simulation → direct-action policy,
compared against Pure Pursuit (PP) and RL-tuned PP [Elgouhary & El-Wakeel, arXiv:2602.18386]
on held-out F1TENTH circuits and synthetic contours.

**Status (2026-11-04):** the code, data split, training runs, test evaluation and result figures in
this repository were produced by an audit-and-correction pass and two later revisions (shared lap
rule, wider PP grids, Bead-PP, five seeds, revised PPO fine-tuning, revised statistics, true-centerline
metrics, DAgger with the Bead-PP teacher, BC-weight check, Stanley and dynamic-model baselines).
Result figures are in `results/figures/`, result tables in `results/final/tables/`. Material from the
earlier version (its figures, data, plots and videos) is in `archive/original_version/` for provenance
only: its "Bead-PPO" model was **not** PPO-trained (expert regression + DAgger) and was trained on its
own evaluation circuits.

## Repository layout
```
docs/                              audit, revision plans and reports, verification report
our_model_code/                    bead environment, Gym wrapper, metrics, geometry; legacy bead scripts
research_paper_code/               Pure Pursuit, RL-PP environment; legacy RL-PP scripts
experiments/                       corrected pipeline: prepare_data, tune_pp, train_bead, train_rlpp,
                                   bead_pursuit, evaluate, analyze, verify_results, ppo_diagnostic,
                                   selection_sensitivity, car_lateral_accel, centerline_metrics,
                                   bc_check, make_figures, run_queue
tests/                             67 automated tests
datasets/
  f1tenth_circuits/                23 public centerlines (commit b95c4ef) + rasterised images
  closed_shapes_dataset/, curve_dataset/   synthetic images (700/150/150 each)
  path_cache/                      extracted reference paths used by every experiment
  split_manifest.json              train/val/test assignment of all 493 references, with reasons
  research_paper_tracks/, test_images/, dataset_generators/   inputs of the original version
runs/                              training outputs per run: config, logs, checkpoints (best.zip used)
  queues/, logs/                   exact launch commands, stdout logs, exit codes
results/
  figures/                         result figures (experiments/make_figures.py)
  final/                           episodes.csv (7,700 test episodes), traj/ (raw trajectories), expert_k/,
                                   centerline/ (true-centerline metrics), tables/, verification/,
                                   controllers.json
  models/                          checkpoints of the original version (re-evaluated in results/final)
archive/original_version/          superseded data, figures, plots, videos and metrics
```

## Reproduce
```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # 67 tests

python experiments/prepare_data.py                          # rasterise, extract, split manifest
python experiments/expert_trainval_probe.py                 # expert on train/val -> v_match
python experiments/tune_pp.py 6                             # Fixed/Adaptive PP and Bead-PP grids (training refs only)
python experiments/bead_pursuit.py 4                        # expert aiming k = 1..10 waypoints ahead

# bead policies, seeds 0-4 (exact commands: runs/queues/q4_dagger.txt, q6_bead_ppo.txt)
python experiments/train_bead.py --mode dagger --seed 0 --workers 2
python experiments/train_bead.py --mode ppo --seed 0 --timesteps 3000000 --eval_every 250000 --log_std_init=-0.5
python experiments/train_bead.py --mode dagger_ppo --seed 0 --timesteps 2000000 --eval_every 200000 \
       --critic_warmup 50000 --log_std_init=-2.0 --lr 1e-4 --bc_coef 1.0
#   add "--reward progress --tag _acc" for the per-progress centering reward
#   DAgger with the tuned Bead-PP controller as teacher (runs/queues/q7_dagger_bpp.txt):
python experiments/train_bead.py --mode dagger --teacher beadpp --tag _bpp --seed 0 --workers 1
#   BC-weight check, lambda in {0.1, 10}, seeds 0-2 (runs/queues/q8_ppo_bc.txt), then:
python experiments/bc_check.py                              # selects lambda on validation only

# RL-PP adaptation of [4], seeds 0-4 (runs/queues/q5_rlpp.txt)
python experiments/train_rlpp.py --mode joint --seed 0 --timesteps 100000 --eval_every 25000
python experiments/train_rlpp.py --mode ld_only --seed 0 --timesteps 100000 --eval_every 25000
python experiments/train_rlpp.py --mode joint --matched --seed 0 --timesteps 100000 --eval_every 25000

python experiments/evaluate.py --workers 16                 # single test-set evaluation
#   controllers added later are evaluated with --only NAME --append (other rows stay unchanged)
python experiments/analyze.py                               # summaries + statistics
python experiments/centerline_metrics.py                    # metrics against the true centerlines
python experiments/ppo_diagnostic.py                        # validation-only reward/noise diagnostic
python experiments/selection_sensitivity.py                 # selection score vs RMSE-only selection
python experiments/car_lateral_accel.py                     # lateral acceleration of the kinematic car
python experiments/verify_results.py                        # integrity checks + re-execution
python experiments/make_figures.py                          # results/figures/
```
The exact queue files used are in `runs/queues/`; per-run exit codes and wall times in
`runs/logs/queue_status.jsonl`. The analysis plan of the revision, written before the final
evaluation, is `docs/03_REVISION_PLAN.md`. Superseded runs (kept for their validation logs) are in
`runs/bead/superseded/`.

## Main test-set result (8 held-out circuits; full tables in `results/final/tables/`)
| Controller | RMSE [cm] | RMSE vs true centerline [cm] | Lap time [s] |
|---|---|---|---|
| Geometric expert (non-learned, bead) | 1.07 | 1.60 | 42.5 |
| Bead-PP: the Fixed PP law on the bead, at v_match | 0.28 | 1.24 | 42.5 |
| Bead-DAgger, expert teacher (5 seeds) | 1.06 | 1.60 | 42.4 |
| Bead-DAgger, Bead-PP teacher (5 seeds) | 0.42 | 1.24 | 42.8 |
| Bead-DAgger→PPO (5 seeds, 2e6 steps, BC term) | 1.77 | 2.26 | 42.2 |
| Bead-DAgger→PPO, BC weight 10 (selected on validation; 3 seeds) | 1.26 | 1.73 | 42.7 |
| Bead-PPO from scratch (5 seeds, 3e6 steps) | 5.12 | 5.49 | 36.9 |
| Fixed PP, gains tuned on training circuits | 0.34 | 1.25 | 18.5 |
| RL-PP joint, adaptation of [4] (5 seeds) | 1.83 | 2.22 | 18.1 |
| Fixed PP tuned, speed matched to bead | 0.31 | 1.23 | 42.2 |

All four pre-declared primary comparisons agree in sign on all 8 circuits (Holm-corrected p = 0.031),
against the extracted references and against the true centerlines.
