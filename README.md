# Direct-Actuation Bead Tracing versus RL-Tuned Pure Pursuit

Image → ordered reference path → 2-D point-mass ("bead") simulation → direct-action policy,
compared against Pure Pursuit (PP) and RL-tuned PP [Elgouhary & El-Wakeel, arXiv:2602.18386]
on held-out F1TENTH circuits and synthetic contours.

**Status (2026-10-11):** the code, data split, training runs, test evaluation and revised paper in
this repository were produced by an audit-and-correction pass. Start with
`paper/Revised_Paper_Bead_Tracing.pdf`, then `docs/REPORT.md` and `docs/VERIFICATION.md`. Material
from the earlier version (original PDF, its tables/figures/videos) is in `archive/original_version/`
for provenance only: its "Bead-PPO" model was **not** PPO-trained (expert regression + DAgger) and
was trained on its own evaluation circuits.

## Repository layout
```
paper/
  Revised_Paper_Bead_Tracing.pdf   the revised paper (10 pages)
  revised/                         LaTeX source: main.tex, tables/ (generated), figures/ (generated)
  scripts/                         make_figures_v2.py, make_tables_v2.py (built from raw records)
docs/                              01_PHASE1_AUDIT, 02_PHASE2_PLAN, REPORT, VERIFICATION
our_model_code/                    bead environment, Gym wrapper, metrics, geometry; legacy bead scripts
research_paper_code/               Pure Pursuit, RL-PP environment; legacy RL-PP scripts
experiments/                       corrected pipeline: prepare_data, tune_pp, train_bead, train_rlpp,
                                   evaluate, analyze, verify_results, ppo_diagnostic, run_queue
tests/                             45 automated tests
datasets/
  f1tenth_circuits/                23 public centerlines (commit b95c4ef) + rasterised images
  closed_shapes_dataset/, curve_dataset/   synthetic images (700/150/150 each)
  path_cache/                      extracted reference paths used by every experiment
  split_manifest.json              train/val/test assignment of all 493 references, with reasons
  research_paper_tracks/, test_images/, dataset_generators/   inputs of the original version
runs/                              training outputs per run: config, logs, checkpoints (best.zip used)
  queues/, logs/                   exact launch commands, stdout logs, exit codes
results/
  final/                           episodes.csv (2,940 test episodes), traj/ (raw trajectories),
                                   tables/, verification/, controllers.json
  models/                          checkpoints of the original version (re-evaluated in the paper)
archive/original_version/          superseded paper, data, figures, plots, videos and metrics
```

## Reproduce
```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # 45 tests

python experiments/prepare_data.py                          # rasterise, extract, split manifest
python experiments/expert_trainval_probe.py                 # expert on train/val -> v_match
python experiments/tune_pp.py 6                             # PP grid search (training refs only)

# bead policies (seeds 0 and 1 were run)
python experiments/train_bead.py --mode dagger --seed 0 --workers 4
python experiments/train_bead.py --mode ppo --seed 0 --timesteps 400000 --eval_every 100000 --log_std_init=-0.5
python experiments/train_bead.py --mode dagger_ppo --seed 0 --timesteps 400000 --eval_every 100000 --critic_warmup 50000 --log_std_init=-0.5

# RL-PP baseline (joint, Ld-only, matched speed)
python experiments/train_rlpp.py --mode joint --seed 0 --timesteps 100000 --eval_every 25000
python experiments/train_rlpp.py --mode ld_only --seed 0 --timesteps 100000 --eval_every 25000
python experiments/train_rlpp.py --mode joint --matched --seed 0 --timesteps 100000 --eval_every 25000

python experiments/evaluate.py --workers 18                 # single test-set evaluation
python experiments/analyze.py                               # tables + pre-declared statistics
python paper/scripts/make_figures_v2.py
python paper/scripts/make_tables_v2.py                      # LaTeX tables + number macros
```
The exact queue files used are in `runs/queues/`; per-run exit codes and wall times in
`runs/logs/queue_status.jsonl`. The DAgger runs were stopped after 6 of 8 rounds (see
`runs/bead/dagger/seed*/NOTE.txt`). Compile the paper with pdfLaTeX (e.g. Overleaf) from
`paper/revised/`. To check the results: `python experiments/verify_results.py`.

## Main test-set result (8 held-out circuits; full tables in the paper)
| Controller | RMSE [cm] | Lap time [s] |
|---|---|---|
| Geometric expert (non-learned, bead) | 1.07 | 42.5 |
| Bead-DAgger (2 seeds) | 1.07 | 42.4 |
| Bead-DAgger→PPO (2 seeds) | 3.31 | 40.8 |
| Bead-PPO from scratch (2 seeds, 4e5 steps) | 7.40 | 50.0 |
| Fixed PP, gains tuned on training circuits | 0.90 | 18.2 |
| RL-PP joint, retrained (2 seeds) | 1.03 | 18.2 |
| Fixed PP tuned, speed matched to bead | 0.77 | 41.6 |

No pre-declared comparison is significant after Holm correction (n = 8 circuits, 2 seeds).
