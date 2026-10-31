# Phase 3 — Revision after review: changes and analysis plan

Written on 2026-10-17, after the reviewer comments and **before** the final test-split evaluation of
any controller trained or tuned under the changes below.

Disclosure of test-split exposure before this plan: (i) every result of Phase 2 (2 seeds, previous
PP grid, previous car lap rule) is known to the authors; (ii) earlier on 2026-10-17 a preliminary
same-plant run (`experiments/bead_pursuit.py`, first version) executed Bead-PP and the Expert-k sweep
once on the test split, with Bead-PP tuned on training references only. Neither result was used to
choose any setting below; every setting below is chosen on training or validation references.

## A. Changes to simulator and baselines
| Change | Reviewer point | Effect |
|---|---|---|
| One lap rule for both plants: the car uses the bead's waypoint counter (`BeadEnvironment.advance_counter`) and episode end (`episode_end`: counter at 99 % of the path, or step cap, or 400 steps without progress) | car ended 6-7 waypoints before closure, bead 3 | Geom@τ end-of-lap artifact removed; bead behaviour bit-identical (verified on stored trajectories) |
| Lower lookahead bound of all PP controllers 0.20 → 0.05 m (`rl_pure_pursuit_env.L_MIN`); classical grids: L_d ∈ {0.05, 0.075, …, 0.20, 0.25, …, 1.00}, g ∈ {0.6, …, 1.4}; adaptive a ∈ {0, 0.025, …, 0.5}, b ∈ {0, 0.025, 0.05, …, 0.25} | tuned optima on the 0.20/0.25 m bound; RL-PP pinned at 0.20 m | optimum must be interior (recorded in `runs/pp_tuning.json`, `on_edge`); RL-PP retrained with action range [0.05, 1.20] m |
| Bead-PP: the car's Fixed PP law on the bead plant at v_match (`common.BeadPursuit`), tuned on the same grid and references as the car | same plant, different controller | new controller |
| Expert-k sweep, k = 1..10, anchored on w_t (Eq. 4) and on the tracked nearest point | lookahead of the expert | new analysis |

## B. Training
* Seeds 0–4 for every learned controller.
* DAgger: the full 8 rounds of Algorithm 1.
* PPO from scratch: 3e6 steps, log_std_init −0.5 (pilot, seed 100, reported in the paper).
* DAgger→PPO: log_std_init −2.0, learning rate 1e-4, behaviour-cloning term (weight 1.0, not tuned)
  towards the expert action recovered from the observation, critic warm-up 5e4 steps, 2e6 steps.
* Reward variant `progress`: the centering bonus is paid per waypoint passed (2.0 × bonus per
  waypoint) instead of per step; all other terms unchanged. Trained for both PPO procedures
  (`*_acc`). Validation diagnostic (expert, 4 validation circuits × 3 noise seeds): with noise
  σ = 0.61 the base reward rises from 746 to 763 although RMSE triples; the progress reward
  falls from 836 to 722.
* RL-PP joint, L_d-only, joint at v_match: 1e5 steps each, VecNormalize as before.
* Checkpoint selection unchanged (`val_score`); its sensitivity to selecting by RMSE alone is
  reported from the validation logs.

## C. Statistics (test circuits, n = 8)
Primary evidence: mean paired difference with a 95 % bootstrap CI (10 000 resamples of circuits,
and of training seeds within learned controllers).

Primary hypothesis tests, metric RMSE, exact two-sided Wilcoxon signed-rank, Holm over these 4:
1. Bead-DAgger vs Fixed PP tuned at v_match (direct actuation vs tuned PP, same speed)
2. Bead-DAgger→PPO vs Bead-DAgger (does reward-based fine-tuning improve imitation)
3. RL-PP joint vs Fixed PP tuned, native speed (learned vs constant PP parameters)
4. Bead-PP vs geometric expert (same plant, different controller)

With n = 8 the smallest attainable p is 0.0078; Holm over 4 tests requires p < 0.0125 for the
smallest, so a test can reject only if all 8 circuits agree in sign. All other comparisons are
reported with bootstrap CIs and unadjusted p as exploratory. Episodes are not pooled: the five
episodes on a circuit differ only by a ±5 cm start perturbation and are not independent replicates.
