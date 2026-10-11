# Verification of the reported results

Date: 2026-10-11. This document answers three questions: (1) do the reported counts reconcile,
(2) are the reported numbers produced by the code and data in this repository, and (3) does the
PPO implementation and evidence support the PPO conclusions. Everything below was produced by
`experiments/verify_results.py` (output: `results/final/verification/report.json`) and
`experiments/ppo_diagnostic.py` (output: `results/final/verification/ppo_diagnostic.json`).

## 1. Counts

### References (paper §IV-A) — the earlier "471" was wrong; corrected to 493
| Split | F1TENTH circuits | Closed shapes | Open curves | Total |
|---|---|---|---|---|
| train | 11 | 60 | 60 | 131 |
| val | 4 | 25 | 25 | 54 |
| test | 8 | 150 | 150 | 308 |
| **all** | **23** | **235** | **235** | **493** |

All 493 were extracted without failure (`datasets/split_manifest.json`, field `extraction_ok`).
Of the 300 synthetic test references, the first 50 per set (100) were evaluated; the other 200 were
extracted but not evaluated (compute budget). "471" in the previous draft was an arithmetic error
in the text, not a property of the data.

### Test episodes (paper §IV-C) — 2,940 is correct; accounting now stated in the paper
21 controller instances × (8 test circuits × 5 start seeds + 100 synthetic contours × 1) = 21 × 140 = **2,940**.

| Instances | Count |
|---|---|
| Single instance: expert; Fixed PP orig./tuned/tuned-matched; Adaptive PP orig./tuned/tuned-matched; published bead checkpoint; published RL-PP checkpoint | 9 |
| Two training seeds each: Bead-DAgger, Bead-PPO, Bead-DAgger→PPO, RL-PP joint, RL-PP Ld-only, RL-PP joint-matched | 6 × 2 = 12 |
| **Total** | **21** |

Checked from `results/final/episodes.csv`: every instance has exactly 140 rows; 8 circuits × 5
seeds {50000, 50137, 50274, 50411, 50548}; 50 closed + 50 open contours with seed 777.

## 2. Are the numbers produced by the code? — yes, all checks pass

| Check | What was done | Result |
|---|---|---|
| B. Record integrity | All 2,940 rows: metrics recomputed from the stored raw trajectory (`results/final/traj/*.npz`) and the stored reference (`datasets/path_cache`) with `metrics.trajectory_metrics` | 0 missing trajectories; max difference ≤ 5e-7 for every metric used in the paper (float32 storage); off-path fraction 2e-4 and peak speed 7e-5 (threshold/rounding effects of float32 storage; neither is reported in the paper's tables) |
| C. Independent code | RMSE and Geom@0.10 recomputed with a separate brute-force point-to-segment implementation (no shared code with `geometry_utils`/`metrics`) | max difference 2.2e-8 (RMSE), 1e-16 (Geom@0.10) |
| D. Re-execution | 12 instances (expert, Bead-DAgger ×2, Bead-PPO ×2, Bead-DAgger→PPO ×2, published bead, Fixed PP tuned, Fixed PP tuned-matched, RL-PP joint ×2) reloaded from their saved checkpoints and re-run on all test circuits × 5 seeds (480 episodes) | **bit-identical trajectories** (max position difference 0.0 m, identical lengths) |
| E. Tables | `paper/scripts/make_tables_v2.py` re-run into a temporary directory from `episodes.csv` | all 5 LaTeX table/number files **byte-identical** to `paper/revised/tables` |
| F. Training runs | exit codes from `runs/logs/queue_status.jsonl` | all 10 final runs exit 0 |

The headline numbers quoted in the paper (e.g. 1.07 cm Bead-DAgger, 3.31 cm Bead-DAgger→PPO,
7.40 cm Bead-PPO, 0.90 cm Fixed PP) are therefore reproduced from raw trajectories, by independent
code, and by re-running the checkpoints. No number in the paper had to be changed as a result of
these checks; the only corrections were the reference count (471 → 493) and the interpretation of
the PPO result (Section 3).

### Checkpoints and training logs per reported seed
| Run | best.zip md5 | env steps | val-log rows | logs |
|---|---|---|---|---|
| bead/dagger/seed0 | f39d5f209e8924ec3aa5f993914c9755 | DAgger, 6 rounds | 6 | `runs/logs/dagger_s0.log`, `NOTE.txt` |
| bead/dagger/seed1 | f04864619412417b9ca5bbc13daac493 | DAgger, 6 rounds | 6 | `runs/logs/dagger_s1.log`, `NOTE.txt` |
| bead/ppo/seed0 | 2c9dcde34824d54e8a8822d2edda7a56 | 401,408 | 5 | `progress.csv` (49 updates), `runs/logs/ppo_s0.log` |
| bead/ppo/seed1 | b1470f9310ccd8f38709dae0dc72b7a5 | 401,408 | 5 | idem |
| bead/dagger_ppo/seed0 | 9a8e7ed1b7b33a1cb99601885fb678b0 | 401,408 | 5 | `progress.csv`, `init.txt` |
| bead/dagger_ppo/seed1 | b8b9f8437fc889dfb0f9372c136428d8 | 401,408 | 5 | idem |
| rlpp/joint/seed0 | ba3cd6915e45fdf490a8d5aed641e506 | 100,352 | 5 | `progress.csv`, `best_vecnorm.pkl` |
| rlpp/joint/seed1 | c93c44af95bd8e57055331ea2f5aab1d | 100,352 | 5 | idem |
| rlpp/ld_only/seed0 | 06758f4a3fd5aef1bfaef244ad636edb | 100,352 | 5 | idem |
| rlpp/ld_only/seed1 | 553fb21d4fbda0a093293d2f5e24db7a | 100,352 | 5 | idem |
| rlpp/joint_matched/seed0 | f11c4cf139bf749b7c250e9a166a63af | 100,352 | 5 | idem |
| rlpp/joint_matched/seed1 | c873c63592f22bb93c3bdfbeb17f7d1c | 100,352 | 5 | idem |
| published PPO_BTP_MODEL.zip | 007c3bc238e82114599e88955f573ad4 | (original; no training log exists) | – | – |
| published RL_PP_JOINT_MODEL.zip | 5af890e9454caa3576c0299affaa8f08 | (original; no training log exists) | – | – |

Each run directory also has `config.json` (all arguments, hyper-parameters, train/val ids).
Classical PP settings: `runs/pp_tuning.json` (full 300-configuration grid with scores) and
`runs/logs/tune_pp.log`. Matched speed: `runs/expert_trainval.json`.

### Exact commands
Training commands are the queue files `runs/queues/q2_ppo_budget.txt`, `q3_rlpp.txt` and the
DAgger lines of `q1_dagger_pilot.txt` (seeds 0–1). Evaluation and analysis:
```
python experiments/evaluate.py --workers 18      # writes results/final/episodes.csv, traj/, controllers.json
python experiments/analyze.py                    # results/final/tables/*
python paper/scripts/make_tables_v2.py           # paper/revised/tables/*
python paper/scripts/make_figures_v2.py          # paper/revised/figures/*
python experiments/verify_results.py             # this verification
python experiments/ppo_diagnostic.py             # Section 3
```
`results/final/controllers.json` lists, for each of the 21 instances, the exact checkpoint and the
controller configuration (mode, gains, speed cap, normalisation statistics).

## 3. PPO: implementation and interpretation

### Implementation checks
* Unit tests (`tests/test_ppo_training.py`, 4 tests): PPO updates actor, critic and log σ; flipping
  the reward sign changes the update (the reward drives learning); checkpoints round-trip;
  deterministic evaluation has no side effects; the DAgger→PPO weight transfer reproduces the DAgger
  actor exactly with a fresh critic, and a frozen-actor update leaves the actor unchanged while
  moving the critic.
* Training logs (`progress.csv`): during the critic warm-up of Bead-DAgger→PPO (6 updates) the
  approximate KL was < 1e-9 (actor frozen as intended). Final approximate KL 0.005–0.016, final
  explained variance 0.71–0.92, action std decreased from 0.60–0.61 to 0.45–0.48. Bead-PPO's training
  progress rose from ~0 to 0.93–0.99 of the reference. Nothing indicates a broken update.

### Diagnostic (validation circuits only)
| Policy (mean of 2 seeds, 4 validation circuits, 3 noise seeds) | Reward | RMSE | Time |
|---|---|---|---|
| Bead-DAgger, deterministic | 746 | 0.87 cm | 48 s |
| Bead-DAgger + PPO's initial noise (σ = 0.61) — PPO's starting policy | 766 | 2.84 cm | 60 s |
| Bead-DAgger→PPO + its own noise — PPO's final policy | 632 | 4.26 cm | 53 s |
| Bead-DAgger→PPO, deterministic | 604 | 3.90 cm | 45 s |

* PPO fine-tuning **lowered** the return of the stochastic policy it optimises (766 → 632; per seed
  766 → 694 and 766 → 571). The earlier draft's explanation ("near-optimal policy moved away by
  noise") was replaced: the result is a failure of fine-tuning to improve its own objective on
  held-out circuits within the budget; the cause was not isolated. The rising training reward in
  `progress.csv` is not evidence of improvement, because each logged mean mixes references of very
  different length.
* The reward is **not monotone in accuracy**: adding noise to DAgger makes it 3× less accurate and
  slower but earns more reward, because the centring bonus is paid every step. The draft sentence
  "the reward and the accuracy metrics agree in ranking" was qualified accordingly, and the
  limitation was added (threats (xi), (xii); future work (b)).

### Are the experiments sufficient for the PPO conclusions?
They support only these scoped statements, which is what the paper now says:
1. With the stated hyper-parameters, 4e5 steps and 2 seeds, PPO from scratch did not reach the
   DAgger/expert accuracy, and PPO fine-tuning degraded the DAgger policy.
2. The reward as designed favours slow traversal and is an imperfect proxy for accuracy.

They do **not** support "PPO cannot learn this task" or "PPO is inferior to imitation in general":
the budget is 7.5× smaller than planned, only one PPO configuration was tested, fine-tuning
hyper-parameters were not tuned, and 2 seeds bound rather than estimate seed variance.

## 4. Failed, stopped and excluded runs (complete record)
| What | Status | Used in results? |
|---|---|---|
| First launch attempt via shell job control (DAgger s0/s3, pilot σ₀=0) | started partially, killed after a few minutes because the launcher dropped jobs; directories deleted before relaunch | no |
| Smoke runs (`*_smoke`, `rlpp/joint/seed99`, seed 99) | short code checks; deleted | no |
| Pilot runs σ₀ ∈ {−1, −0.5, 0}, seed 100 | stopped at ~7.5e5 of 3e6 steps (budget); used only to select σ₀ = −0.5 on validation | selection only; never reported as results |
| DAgger seeds 0, 1 | stopped after round 6 of 8 (budget; validation score flat since round 2) | yes (`NOTE.txt`) |
| DAgger seeds 2–4 | queued, never started (queue stopped for the budget) | no |
| PPO / DAgger→PPO seeds 2–4, RL-PP seeds 2–4 | not run (budget) | no |
| Queue runner q1 | killed deliberately (reported exit 255) | – |
| Final runs (10) | all exit 0 | yes |
| Test synthetic references 51–150 per set | extracted, not evaluated (budget) | no |
| Test episodes | none excluded; all 2,940 rows are reported and enter the tables | yes |
| Image extraction failures | none (493/493) | – |
