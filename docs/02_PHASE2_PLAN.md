# Phase 2 — Correction plan and pre-declared analysis

Written on 2026-10-11 before any test-split evaluation of a new or retrained controller.
(The only test-circuit executions before this point are the audit re-runs of the *published*
checkpoints and of the expert on the three paper circuits, recorded in `01_PHASE1_AUDIT.md`.)

## A. Changes to the core model (smallest set)
| Change | Why | Effect on comparability |
|---|---|---|
| Per-instance RNG; `reset(seed=)` honoured | Global `np.random.seed` side effects (audit #9) | Start-perturbation draws differ from the original runs (same ±5 cm distribution) |
| `strict=True` path loading in all experiments | Silent default-circle fallback (#10) | None for valid images |
| Step cap = max(3500, ⌈1.5·L/(v*·Δt)⌉), same cap for PP | 3500 steps cannot cover 82 m at v* (#11) | Unchanged for completed runs; only lets slow-but-progressing runs finish |
| PP env: optional `v_cap`; parameterised adaptive schedule | Controlled comparison (#4), fair baseline tuning (#8) | Defaults reproduce the original behaviour |

Unchanged: dynamics (Eq. 1), reward weights (Table I), 16-D observation, action meaning and
scaling, image→path front end, network architecture.

## B. Training (affects experiments)
* `dagger` — Alg. 1, unchanged schedule/learning rates, on the training split; best-on-validation
  checkpoint kept (original code overwrote it, #5); fixed 8 rounds (the original early-stop rule
  was defined on the evaluation circuits).
* `ppo` — genuine PPO on the env reward, Table II "network configuration" hyper-parameters
  (lr 3e-4, n_steps 2048/env, batch 128, 10 epochs, γ 0.99, λ 0.95, clip 0.2, ent 0.001, vf 0.5,
  grad-norm 0.5, separate [256,256] tanh). Only addition: `log_std_init`, chosen from
  {−1.0, −0.5, 0.0} by a validation-only pilot with a seed (100) that is never reported.
* `dagger_ppo` — actor from the same-seed `dagger` best checkpoint; critic warm-up with the actor
  frozen (300k steps); then PPO with the same settings and the same total step budget as `ppo`.
* RL-PP joint and a separately trained Ld-only policy, 5 seeds, original Table II recipe, on the
  training split, with VecNormalize statistics saved and applied (newly found defect: the
  original script trained on normalised observations but evaluated on raw ones).
* Fixed PP and Adaptive PP grid-tuned on training references (native and matched speed).
* Seeds 0–4 for every learned controller.

## C. Data split (fixed before training; `datasets/split_manifest.json`)
* Circuits (23 public F1TENTH centerlines, commit b95c4ef): test = Hockenheim, Montreal,
  YasMarina (paper) + 5 by seeded permutation; val = 4; train = 11.
* Synthetic: train 60+60, val 25+25, test 150+150 (i.i.d. from the same generator — an
  in-distribution test, not a geometry-family hold-out).

## D. Metrics (one implementation for all controllers, `our_model_code/metrics.py`)
Geom@{0.05,0.10,0.15}, RMSE, P95, max, off-path fraction (e > 0.10 m), strict progress (bead
pointer rule), ordered coverage (DP), completion (ordered coverage ≥ 0.95), traversal time,
mean/peak speed, RMS acceleration, boundary contacts, and the native action-change measure.
Completion threshold and ordered coverage were defined after a validation smoke run showed that
the strict pointer stalls permanently after one excursion; no test result had been seen.

## E. Evaluation protocol
* Test circuits: 5 episodes each, start seeds 50000+137k (as in the paper).
* Test synthetic: 1 episode each, seed 777.
* Deterministic policies (actor mean); no gradient updates during evaluation.
* Track A: every controller in its own plant at its own speed.
* Track B: PP family with the reference speed capped at v_match = 1.256 m/s (mean expert speed on
  training circuits), PP gains re-tuned at that speed, RL-PP retrained at that speed.

## F. Pre-declared comparisons (test circuits; primary metrics RMSE and Geom@0.10)
1. dagger_ppo vs dagger  2. ppo vs dagger_ppo  3. dagger_ppo vs expert
4. dagger_ppo vs Fixed PP tuned (Track A)  5. dagger_ppo vs Fixed PP tuned matched (Track B)
6. RL-PP joint vs RL-PP Ld-only  7. RL-PP joint vs Fixed PP tuned.

Unit of analysis: the test circuit (n = 8), value = mean over training seeds and evaluation
episodes. Test: exact two-sided Wilcoxon signed-rank; effect: mean paired difference with a 95 %
bootstrap CI (10 000 resamples of circuits, and of training seeds within learned controllers).
Holm correction over the 14 tests. With n = 8 the smallest attainable two-sided p is 0.0078, so
"no significant difference" is expected to be common and will be reported as inconclusive, not as
equivalence. The same comparisons on the 300 synthetic references are reported as secondary.

## G. Budget-driven deviations (recorded 2026-10-11, before any test evaluation)
The user limited the remaining compute to about 30 CPU-minutes. Changes to the plan above:
* Training seeds: **2** (0, 1) instead of 5 for every learned controller. Training-seed variance
  is therefore only indicative; no between-seed statistics are claimed.
* DAgger: stopped after 6 of 8 rounds (validation score flat from round 2 onward).
* PPO pilot: stopped at ~750k of 3M steps; `log_std_init = -0.5` selected (best validation score).
* PPO and DAgger->PPO: 400k environment steps each (planned 3M); critic warm-up 50k steps.
  PPO-from-scratch is therefore heavily under-trained relative to the pilot curve.
* RL-PP: 100k steps per run (original recipe used 60k on one contour), seeds 0-1.
* Test synthetic references: first 50 per family (of 150). Test circuits unchanged (8 x 5 episodes).
* Statistics: the circuit-level Wilcoxon/bootstrap analysis is kept (unit = circuit, n = 8);
  bootstrap resamples the 2 seeds only as a nuisance factor.
