# Final report — audit, correction, experiments and revised paper

> **Superseded in part (2026-10-17).** The results below describe the state before the revision after review
> (2 seeds, previous PP grid and lap rule). For the current results see `04_REVISION_REPORT.md`,
> `03_REVISION_PLAN.md` and `results/final/verification/report.json`.


Date: 2026-10-11. Companion documents: `01_PHASE1_AUDIT.md` (audit before any change),
`02_PHASE2_PLAN.md` (plan, pre-declared analysis, budget deviations). Compute used for new
training + evaluation after the user's budget instruction: about 30 CPU-minutes, plus earlier
partial runs (DAgger, pilot) that were stopped.

---
## 1. Executive audit — VERIFIED AND TESTED
1. **"Bead-PPO" was never trained with PPO.** `train_ppo_v9.py` never calls `learn()`; the actor was
   regressed onto a geometric expert (behaviour cloning + DAgger). Critic, action noise and reward
   were unused. The paper disclosed this in §III-E but named, framed and indexed the method as PPO.
2. **Train = test.** The policy was trained and checkpoint-selected on the three circuits it was scored on.
3. **The strongest baseline was missing.** The non-learned expert the policy imitates was more
   accurate than the policy and than every PP variant on those circuits.
4. **Baselines were handicapped.** Fixed PP gains were untuned (RMSE 3.89 cm vs 0.90 cm tuned);
   the RL-PP policy was trained on normalised observations but evaluated on raw ones (5.00 cm vs
   1.03 cm when corrected).
5. Further defects: the best checkpoint was overwritten by the last one; global-RNG seeding; a silent
   default-circle fallback; a step cap too short for long circuits; two different "sequential
   coverage" definitions in one table; placeholders and an incorrect variability statement in the PDF.
6. The **published numbers are genuine** — the stored checkpoints reproduce Tables III/IV exactly.

**Corrected conclusion:** on 8 held-out circuits the DAgger bead equals its expert (1.07 cm);
genuine PPO made the bead worse (3.31 cm fine-tuned, 7.40 cm from scratch at 4e5 steps); tuned
Fixed PP is more accurate (0.90 cm) at 2.3x the pace, and more accurate still at matched speed
(0.77 cm). The earlier claim of a direct-actuation advantage is not supported. No pre-declared
comparison is significant after Holm correction (n = 8 circuits, 2 seeds).

## 2. Discrepancy table (final status) — VERIFIED AND TESTED
Numbering follows `01_PHASE1_AUDIT.md`; #22 was found during the work.

| # | Issue | Sev. | Correction | Verification | Final status |
|---|---|---|---|---|---|
| 1 | Not PPO-trained | C | `experiments/train_bead.py`: `dagger`, `ppo`, `dagger_ppo` separated and named by procedure; genuine SB3 PPO on reward Eq. (3) | `tests/test_ppo_training.py` (actor+critic+log_std change; reward sign changes the update; KL/clip/value-loss logged); 4 PPO runs completed | corrected and tested |
| 2 | Train = test | C | circuit-level split (11/4/8), paper circuits in test; selection on validation only | `datasets/split_manifest.json`; trainers receive only train ids; `config.json` lists ids | corrected and tested |
| 3 | Expert baseline missing | C | expert evaluated everywhere | Table III of revised paper | corrected and tested |
| 4 | Speed/dynamics confound | Ma | Track B: PP speed capped at v_match=1.256 m/s (from training circuits), PP re-tuned, RL-PP retrained | test results | corrected and tested (holonomic vs bicycle difference remains) |
| 5 | Checkpoint overwritten; save rule ≠ Alg. 1 | Ma | fixed in `train_ppo_v9.py`; new trainers keep best-on-validation | code inspection | corrected, not unit-tested (legacy script not re-run) |
| 6 | Two sequential-coverage rules | Ma | one post-hoc implementation (`metrics.py`): strict progress + ordered coverage (exact DP) | `tests/test_metrics.py` incl. brute-force check | corrected and tested |
| 7 | Single model | Ma | 2 training seeds (budget) | per-seed tables | partially resolved |
| 8 | Baseline tuning unknown | Ma | grid search on training refs (`tune_pp.py`), original gains also reported | `runs/pp_tuning.json` | corrected and tested; optima lie on grid/clip boundary |
| 9 | Global RNG | Mo | per-instance `Generator`, `reset(seed)` honoured | `test_seeding_is_local_and_reproducible` | corrected and tested |
| 10 | Silent circle fallback | Mo | `strict=True` raises `PathExtractionError` | `test_strict_mode_raises` | corrected and tested |
| 11 | Step cap too short | Mo | cap = max(3500, ⌈1.5L/(v*Δt)⌉), same for PP | `test_step_cap_scales_with_length`; expert completes all 185 train/val refs | corrected and tested |
| 12 | Extraction error unreported | Mo | measured vs. source centerline for 23 circuits (0.79–1.24 cm mean, ≤6.53 cm max) | `prepare_data.py`, manifest | resolved (reported) |
| 13 | Reward untested / possible exploitation | Mo | per-episode reward components logged; expert has the highest reward, PPO policies lower → no reward hacking observed, PPO under-optimised | `episodes.csv` rc_* columns | verified; reward weights unchanged |
| 14 | Absolute-position features permit memorisation | Mo | evaluated on held-out circuits; observation unchanged | — | not tested by ablation |
| 15 | Synthetic test is i.i.d. | Mi | stated in paper | — | resolved (disclosed) |
| 16 | "<0.3 pp" variability claim false | Mi | removed | stored records | corrected |
| 17 | PDF placeholders | Mi | Table IV footnote, [gmin]/[Lmin] filled; author block left for authors | — | partially resolved (author details pending from user) |
| 18 | README conflict markers; mislabelled checkpoint | Mi | README rewritten; RL-PP trainer no longer writes `COMPLETE_BTP_MODEL` | — | corrected |
| 19 | Unpinned deps; CWD outputs | Mi | `requirements.txt` pinned; new scripts write under `runs/`, `results/` | — | corrected |
| 20 | Dynamics/reward/obs/pipeline vs paper | — | none needed | code inspection, `test_environment.py`, `test_image_pipeline.py` | verified |
| 21 | References | — | [4],[5],[22] checked on arXiv; others not fetched | — | partially verified |
| 22 | RL-PP VecNormalize stats never saved/applied | Ma | stats saved with checkpoints and applied (`train_rlpp.py`, legacy trainer fixed) | retrained RL-PP 1.03 cm vs published 5.00 cm | corrected and tested |

## 3. Change log — VERIFIED AND TESTED
Modified: `our_model_code/environment.py` (RNG, strict mode, path pools, step cap, `hit_boundary`
info); `our_model_code/bead_gym_env.py` (seed forwarding, path pools); `our_model_code/train_ppo_v9.py`
(docstring states it is not PPO; checkpoint bug); `research_paper_code/rl_pure_pursuit_env.py`
(RNG, path pools, `v_cap`, adaptive coefficients, common step cap); `research_paper_code/train_rl_pure_pursuit.py`
(VecNormalize save/apply; no mirrored `COMPLETE_BTP_MODEL`); `README.md`; `requirements.txt`.
New: `our_model_code/metrics.py`; `experiments/*`; `tests/*`; `paper/scripts/make_figures_v2.py`,
`make_tables_v2.py`; `paper/revised/*`; `datasets/f1tenth_circuits/*`, `path_cache/`,
`split_manifest.json`; `docs/*`. Unchanged: dynamics, reward weights, observation, action scaling,
image front end, network architecture. `git diff` against commit b85c258 shows every edit.

## 4. Final model description — VERIFIED AND TESTED
Bead: point mass, Eq. (1), a_max=10 m/s², µ=7 s⁻¹, Δt=0.016 s, v*≈1.27 m/s. Observation 16-D,
action 2-D acceleration in [−1,1]². Reward Eq. (3), Table I weights. Network: SB3 actor–critic,
separate [256,256] tanh MLPs, diagonal Gaussian, log σ₀=−0.5. Three procedures (DAgger / PPO /
DAgger→PPO) as in the revised §III-E. The best-performing learned bead policy is **Bead-DAgger**
(imitation, not RL). PPO is implemented and genuinely optimises the policy, but under the available
budget it did not improve accuracy.

## 5. Corrected implementation — VERIFIED AND TESTED
In this repository (see change log). Runnable end to end; commands in `README.md`.

## 6. Test report — VERIFIED AND TESTED
`python -m pytest tests -q` → **44 passed** (16.6 s). Coverage: metric definitions on toy
trajectories (incl. exact brute-force check of ordered coverage), terminal speed and diagonal speed
of Eq. (2), action clipping, wall contact, reward signs, completion bonus, seeding isolation and
reproducibility, path-pool sampling, step cap, strict-mode failures, image pipeline (circle, line
orientation/y-inversion, arc-length uniformity, two-stroke path break, inverted polarity), PPO
update of actor/critic/log σ, reward dependence of the update, checkpoint round-trip, side-effect-free
evaluation, exact Wilcoxon vs scipy, Holm. Failures during development: 3 metric tests had wrong
expectations (corrected after checking the definition); one hypothesis in the draft paper (cause of the
PP Geom@0.10 gap) was tested, found false and replaced by the verified cause (episode termination).
Not unit-tested: the legacy `train_ppo_v9.py` fix, the end-to-end trainers (exercised by real runs instead).

## 7. Experiment report — VERIFIED AND TESTED (with reduced scope)
* Runs: DAgger seeds 0–1 (6/8 rounds, stopped); PPO and DAgger→PPO seeds 0–1 (4e5 steps); RL-PP
  joint / Ld-only / joint-matched seeds 0–1 (1e5 steps); PP grid tuning (300 configurations);
  log σ₀ pilot (seed 100, stopped at 7.5e5 steps, not reported as a result). All 10 final runs exit 0
  (`runs/logs/queue_status.jsonl`).
* Test evaluation: 2940 episodes, 21 controller instances, 8 circuits × 5 start seeds + 100 shapes.
  Raw: `results/final/episodes.csv`, `results/final/traj/*.npz`. Summaries and statistics:
  `results/final/tables/`.
* Independent check: RMSE of five controllers recomputed directly from raw trajectories matches the
  paper tables exactly (1.07, 1.07, 3.31, 0.90, 1.03 cm).
* Not performed (budget): seeds 2–4; full 3e6-step PPO; DAgger rounds 7–8; synthetic test
  contours 51–150; observation/reward ablations; bead at higher speed (raised a_max).

## 8. Reproducibility package — IMPLEMENTED BUT NOT FULLY TESTED
Pinned requirements, configs (`runs/**/config.json`), queues, split manifest, seeds, checkpoints,
raw records, metric/plot/table scripts, tests, README. Not tested: a clean re-install on another
machine, or bit-identical re-training (multi-process rollouts may differ in floating-point order).

## 9. Revised paper — IMPLEMENTED BUT NOT FULLY TESTED
`paper/revised/main.tex` (+ `main.pdf`, 9 pages, compiled with Tectonic 0.15). Same title, IEEE
conference format, section structure (I–VI, same subsections), writing style and figure style as the
original. All tables and quoted numbers are generated from raw records (`make_tables_v2.py`,
`numbers.tex`). Pending: author affiliations/emails (user to supply); compile with pdfLaTeX/Overleaf
to obtain the Times font of the original (the local engine used Computer Modern; page count can
only stay ≤ 9).

## 10. Final integrity checklist
| Item | Status |
|---|---|
| Every reported number traceable to executed rollouts or logs | VERIFIED AND TESTED |
| No hard-coded, estimated, or copied results in outputs | VERIFIED AND TESTED (tables generated by script; old numbers appear only as labelled "published checkpoint" rows re-executed now) |
| Failed / unfavourable results reported (PPO worse; budget truncations) | VERIFIED AND TESTED |
| Test split untouched during training, selection, tuning | VERIFIED AND TESTED (author prior knowledge of the 3 paper circuits disclosed) |
| PPO genuinely used where PPO is claimed | VERIFIED AND TESTED |
| Statistical claims match analysis (none significant after Holm) | VERIFIED AND TESTED |
| Training-seed robustness | NOT VERIFIED (2 seeds only) |
| PPO at full budget (3e6 steps), 5 seeds | NOT PERFORMED |
| Ablations of observation features / reward groups | NOT PERFORMED |
| Physical-robot experiments | NOT PERFORMED |
| References [1]–[3], [6]–[21] fetched and checked | NOT VERIFIED (standard works; details consistent with the published versions as far as known) |

---
## Addendum — repository-wide error check (2026-10-11, after the report above)
* All 58 Python files compile; `pyflakes` reports **0 errors** (was 12 undefined-name errors:
  `Optional`/`Tuple`/`Dict`/`Any` missing imports in `rl_pure_pursuit_env.py`,
  `benchmark_paper_controllers.py`, `plot_paper_interpretability.py`).
* Legacy entry points smoke-run with outputs redirected (9 scripts, 1 episode each): all exit 0 after fixes.
  - `evaluate_complete_v8.py` crashed by default: its default model `COMPLETE_BTP_MODEL.zip` is the RL-PP
    checkpoint (8-D observations). Default changed to `results/models/PPO_BTP_MODEL.zip`.
  - `evaluate_complete_v8.py` and `benchmark_paper_controllers.py` silently overwrote stored result files
    (`results/metrics_and_logs/complete_model_summary.json`, `benchmark_summary.json`). Both files were
    restored from git; the scripts now take `--json` / write next to `--csv`.
* `paper/scripts/make_figures_v2.py`: doubled line breaks introduced by an encoding repair were removed.
* Paper: 0 LaTeX errors, 0 overfull lines, all citations/labels/number macros resolve; Times text font
  (TeX Gyre Termes under XeTeX, IEEEtran Times under pdfLaTeX) matching the original. 9 pages.
  Copy at `paper/Revised_Paper_Bead_Tracing.pdf`.
* Tests: 44 passed.
* Not changed (intentional): untracked file `h -u origin main --force-with-lease` (accidental git log
  output; delete if unwanted); `__pycache__/*.pyc` files are tracked in git and change whenever code runs.

## Addendum 2 — verification pass (see docs/VERIFICATION.md)
* Reference count corrected in the paper: 493 (not 471). Episode count 2,940 confirmed and its
  accounting (21 instances × 140) added to §IV-C.
* All results re-verified: recomputation from raw trajectories, independent metric code,
  bit-identical re-execution of 12 checkpoints, byte-identical table regeneration.
* PPO interpretation corrected after a validation-only diagnostic (`experiments/ppo_diagnostic.py`):
  fine-tuning lowered the return of its own stochastic policy; the reward favours slow traversal.
  §V-B, threats (xi)–(xii), conclusion and future work updated. No reported metric changed.
* New test: DAgger→PPO weight transfer and actor freezing (45 tests pass).
* `results/final/controllers.json` now records repository-relative checkpoint paths.

## Addendum 3 — repository clean-up
* Deleted: stray `h -u origin main --force-with-lease`; all `__pycache__`/`.pytest_cache` (50 `.pyc`
  files removed from git tracking); `datasets/curve_dataset.zip` (stale partial copy, 241 of 1000
  files); `results/models/COMPLETE_BTP_MODEL.zip` (byte-identical to `RL_PP_JOINT_MODEL.zip`);
  `last.zip` of the four PPO runs (weights identical to `final.zip`); pilot-run checkpoints (only
  used for σ₀ selection; logs kept); LaTeX build files; duplicate `paper/figures_v2/`.
* Moved to `archive/original_version/`: original PDF, `paper/data`, `paper/figures`, original paper
  scripts (paths updated), `results/plots`, `results/videos`, `results/metrics_and_logs`, root
  `benchmark_summary.json`.
* Figures are now generated directly into `paper/revised/figures/`. `.gitignore` added.
* After the move: tables regenerate byte-identically, 45 tests pass, paper compiles (10 pages).

## Addendum — revision of 2026-10-24 (IJCAS manuscript)
* Fig. 4 (bars): added "Bead-DAgger, Bead-PP teacher" (0.42 cm, five seeds 0.41-0.43 cm).
* Fig. 8 right panel: legend now "anchored on w_t (Eq. 3)". All figure legends now use the table row
  names (e.g. "Bead, naive protocol", "prog. rew.", "Fixed PP (tuned)", "Geometric expert (3)").
  The tables print the expert's equation number with \ref{eq:expert}; they previously said (4),
  the number of the earlier manuscript.
* Table 8: true-centerline difference and 95% bootstrap interval added for every row, from
  results/final/centerline/comparisons.csv (recomputed from stored trajectories, paired over
  circuits and seeds). Primary rows: +0.36 [+0.22,+0.51], +0.66 [+0.39,+0.97], +0.97 [+0.55,+1.49],
  -0.36 [-0.49,-0.23] cm; identical to the values computed from rounded Table 5 means.
* Table 1: Spielberg's 5.6 cm (and Austin's 10.5 cm) minimum is the final segment of a loop that the
  extraction leaves 8.2 cm (4.6 cm) short of closure; not a junction (0 breaks). Footnote added;
  minimum of the other segments 15.7 (12.9) cm. Values unchanged.
* Stanley baseline: research_paper_code/rl_pure_pursuit_env.py mode "stanley"; tuned with the PP
  protocol (experiments/tune_stanley.py, 153 settings after widening an edge optimum; plateau at
  matched speed); test results in Tables 4, 5, 8. Front/rear-axle diagnostic: front_axle_check.py.
* Dynamic model check: research_paper_code/dynamic_single_track.py (F1TENTH Gym single-track model,
  commit 4fdb9c7; steering limit kept at 1.10 rad); experiments/dynamic_check.py; two variants
  (tyres only / plus 3.2 rad/s steering-rate limit). Reported in Section 5.4 and threats.
* All earlier episode rows, summaries and comparisons are unchanged (checked); 62 tests pass.

## Addendum — paper frozen for submission (2026-10-24, evening)
* Dynamic check rerun with the car's position at the rear axle (was the centre of gravity): tyres
  only, Fixed PP 3.95 cm (88% laps), RL-PP 5.68 cm (73%); Fixed PP ahead on average by 1.73 cm
  [-0.02, 3.38], RL-PP better on 2/8 circuits (earlier CG-based text said "all eight"; corrected).
  With the rate limit: Fixed PP 0 laps, RL-PP 2%. Previous CG-based outputs kept in scratch only.
* Austin closure check: closing the open loop changes RMSE by <= 0.012 cm, no lap outcome
  (results/final/closure_check.json).
* Abstract now states the Stanley and dynamic-model findings; threats list what is not yet tested.
* Still running / not in the paper: retuning on the dynamic car, RL-PP retrained on it, real-scale
  circuits, fast bead, sensor noise, dense waypoints, rear-wheel feedback, 150 contours per set,
  lambda=10 seeds 3-4.

## Addendum — supervisor comments of 2026-10-30 (IJCAS manuscript)
* Figures: the PDFs in paper/ were checked pixel-for-pixel against the source files and captions; no
  mismatch found there. The reported mismatch and the "Eq. 4" legend match older figure files, so
  paper/ijcas_submission(.zip) was added as a self-contained package (tables + figures next to the .tex).
* Dynamic model: Fixed PP and Stanley re-tuned on (a) tyres only and (b) tyres + 3.2 rad/s rate limit
  with the kinematic study's grids, training references and score (tune_extra.py), RL-PP retrained;
  8 test circuits x 5 episodes against extracted references and true centerlines (robustness_eval.py,
  dynamic_stats.py). Fixed PP's gain lies on the shared grid's top (g = 1.4); in (b) no shared-grid
  setting completes every training lap. Gains above 1.4 tested separately. New Table 7, Section 5.4.
* Stanley grid: "flat for k >= 64" was disproved by extending the grid. Widened until interior:
  (512, 0.01) native, (1024, 1) matched. Test: 4.12 and 3.56 cm. Text corrected; all rows regenerated.
* Instance count now taken from episodes.csv (60); verification refreshed (8400 rows, 22 re-executed).
* Fig. 3: Stanley bar added. Algorithm 1 moved after Section 3.3 text. Abstract 233 words.
* Not in the paper: rear-wheel feedback test evaluation, true-scale circuits, fast bead, noise, dense
  waypoints, 150 contours per set, lambda = 10 seeds 3-4.

## Addendum — all figures restored to the full manuscript (2026-10-31)
main_full.tex: 11 figures and 9 tables again (5 figures and the synthetic-contour table back from the
supplement, all figures at column width); 11 pages. Only redundant wording was cut, each part being
stated elsewhere. main.tex and supplementary.tex unchanged.

## Addendum — text overlap reduced (2026-10-31)
About 90 passages reworded in the IJCAS manuscripts; numbers, citations and equations unchanged.
Own proxy (5-word runs against the original paper, the public GitHub paper, ref. [4] and 7 cited arXiv
papers): full manuscript body 1.5%, 9.1% with references. Not a Turnitin result.

## Addendum: supervisor comments on the dynamic-model result and contributions (2026-11-01)

No experiment was re-run; all numbers come from existing result files.

- Main dynamic Fixed PP result is now the inside-range optimum from the extended gain grids: (0.175, 2.5) with 1.83 cm in (a) and (0.45, 1.6) with 3.09 cm in (b). The shared-grid result, whose gain sits on the top edge (1.91 / 3.23 cm), stays in the dynamic table as a secondary row.
- Comparison macros now use `fixed_retuned_highg` as A. Against Stanley: (a) -0.82 cm [-1.44, -0.10], better on 7/8, exact Wilcoxon p=0.078; (b) -0.49 cm [-1.22, 0.18], better on 6/8, p=0.195. Against retrained RL-PP: (a) -3.42 cm [-4.96, -2.22]; (b) -3.64 cm [-5.90, -2.06], both better on 8/8.
- Stanley wording: clear with tyres only, unclear with the rate limit. "Clear" rests on the interval excluding zero; the Wilcoxon p in (a) is above 0.05.
- Data Availability now gives the GitHub URL. Contributions gain a Stanley and dynamic-model bullet; the PP finding is restricted to the kinematic car and the bead.
- `main_full` is now 12 pages (11 figures); `main` 9 pages.

Later on 2026-11-01: overlap rewording round, wording only. Numbers, citations, equations, math, figures and the reference list were checked unchanged. Overlap proxy (5-word runs): `main_full` body 0.5%, 8.2% with references; `main` body 0.3%, 9.1% with references; supplement 0.0%. `main_full` is back to 11 pages.
