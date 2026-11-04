# Phase 1 — Audit of the original paper and implementation

Date: 2026-10-11. Recorded **before** any code was modified. Statuses here are the
state at audit time; the final status of every item is in `docs/REPORT.md`.

## 1. What exists

| Item | Location | Notes |
|---|---|---|
| Paper (8 pp, IEEE template, PDF only — no LaTeX source) | `paper/Dileep_Sanjay_IEEE_Conference_Template__Copy_.pdf` | |
| Bead environment (dynamics, reward, obs, image→path) | `our_model_code/environment.py`, `bead_gym_env.py` | |
| "Bead-PPO" trainer | `our_model_code/train_ppo_v9.py` | expert anchoring + DAgger only |
| RL-tuned Pure Pursuit (re-implementation of [4]) | `research_paper_code/rl_pure_pursuit_env.py`, `pure_pursuit.py`, `train_rl_pure_pursuit.py` | genuine SB3 PPO (`model.learn`) |
| Paper experiment / figure scripts | `paper/scripts/*.py` | figures are drawn from stored rollouts |
| Raw records behind Tables III/IV | `paper/data/summary.json`, `shapes.json`, `trace_*.npz` | |
| Checkpoints | `results/models/*.zip` | `COMPLETE_BTP_MODEL.zip` is byte-identical to `RL_PP_JOINT_MODEL.zip` (md5 5af890e9…) |
| Synthetic datasets | `datasets/closed_shapes_dataset`, `datasets/curve_dataset` (700/150/150 each) | |
| Circuits | `datasets/research_paper_tracks` (3 F1TENTH centerlines) | |
| Training logs for the published checkpoints | **none** | provenance of `PPO_BTP_MODEL.zip` and `RL_PP_JOINT_MODEL.zip` cannot be reconstructed |

Environment found: Python 3.14.3, torch 2.11.0, stable-baselines3 2.8.0, gymnasium 1.2.3,
numpy 2.4.4, opencv-python 4.13.0.92, 22 CPU cores, no GPU used, no LaTeX toolchain.

## 2. Verification runs performed during the audit

1. **Reproduction of the published numbers.** `PPO_BTP_MODEL.zip` and `RL_PP_JOINT_MODEL.zip`
   were re-executed with `paper/scripts/run_paper_experiments.py` functions (seed 50000).
   Results matched `paper/data/summary.json` (e.g. Hockenheim Bead-PPO Geom@0.10 = 0.9914,
   RMSE = 0.0246 m; Montreal 0.9600 / 0.0347; Yas Marina 0.9543 / 0.0409). Tables III and IV in
   the PDF match the stored records exactly. **The published numbers are genuine outputs of the
   stored checkpoints.**
2. **Geometric expert (Eq. 4) run as a controller** on the same plant, same seed:

   | Circuit | Expert Geom@0.10 | Expert RMSE | Bead-PPO (published) RMSE | Best PP RMSE |
   |---|---|---|---|---|
   | Hockenheim | 99.1 % | 0.97 cm | 2.46 cm | 3.62 cm |
   | Montreal | 99.4 % | 1.24 cm | 3.46 cm | 5.74 cm |
   | Yas Marina | 99.1 % | 1.39 cm | 4.06 cm | 5.22 cm |

   The non-learned teacher beats the learned policy and every PP variant. The paper never reports it.
3. **Image→path pipeline vs. ground truth** on all 23 public F1TENTH circuits (rasterised with the
   repository's own `create_track_images.py`; the 3 paper images are regenerated bit-identically):
   mean reference-extraction error 0.8–1.2 cm, max 2.3–6.5 cm, no breaks, 22/23 closed
   (Austin and Spielberg end 4.6 and 8.2 cm short of closure). Path lengths 39–82 m.
4. **Per-seed variation claim.** Paper: "coverage entries vary by less than 0.3 pp". Stored
   records: the max–min range reaches 0.86 pp (Montreal Bead-PPO, Yas Marina Ld-only).

## 3. Discrepancy table (audit-time status)

Severity: C = critical, Ma = major, Mo = moderate, Mi = minor.

| # | Paper claim | Actual implementation / evidence | Issue | Sev. | Required correction | File / section | Verification procedure | Status at audit |
|---|---|---|---|---|---|---|---|---|
| 1 | Title/abstract/name "Bead-PPO", Fig. 1 "Learning (PPO + DAgger)", index term PPO | `train_ppo_v9.py` never calls `model.learn()`; only an MSE regression of the actor mean onto expert labels (BC + DAgger). Critic and `log_std` are never trained; reward is never used. Paper §III-E admits this in the body. | Method is imitation learning, not PPO. Name and framing overstate it. | C | Implement real PPO (rollouts, GAE, clipped surrogate, critic, entropy) driven by the env reward; keep DAgger as a separately named variant/initialiser; rename accordingly | `train_ppo_v9.py`; paper title, abstract, §I, §III-E, Fig. 1 | Code inspection; unit test that PPO updates change actor+critic and that reward changes the gradient | unresolved |
| 2 | Circuit results evaluate Bead-PPO | Bead-PPO trained and checkpoint-selected on Hockenheim/Montreal/Yas Marina — the evaluation circuits (disclosed in §V-F i–ii) | No independent test set; Table III is training/selection performance | C | Circuit-level train/val/test split; paper circuits held out from all training and selection | data, §IV-A, Table III | Split manifest; trainer only receives train paths; selection only on val | unresolved |
| 3 | "Direct-actuation policy tracks more precisely than PP" | Non-learned geometric expert on the same bead plant beats Bead-PPO and all PP (table above) | Advantage attributable to plant/expert, not learning; strongest baseline omitted | C | Report the expert as a baseline in every table | §V, Tables III/IV | Re-run expert under final protocol | unresolved |
| 4 | Bead vs PP comparison | Bead ≈1.3 m/s (damping limit) vs PP 1.8–5 m/s; holonomic vs bicycle (disclosed §V-B) | Speed/dynamics confound; no controlled comparison | Ma | Track B: PP run at matched speed, PP gains re-tuned at that speed on training data | `rl_pure_pursuit_env.py`, §V | Matched-speed runs | unresolved |
| 5 | Alg. 1 "checkpoint if the minimum improves … return best checkpoint" | Save rule is `min_geom > best or mean_geom > 0.96` (can save a worse model and lower `best`); after the loop the **final** model is saved unconditionally over the best one (`train_ppo_v9.py:292-293`) | Code does not implement Alg. 1; the stored checkpoint may be the last round, not the best; no logs to tell | Ma | Fix the checkpoint logic; select on validation data; keep logs | `train_ppo_v9.py` | Test that best ≠ last is preserved | unresolved |
| 6 | Table III "Sequential [%]" compared across controllers | Bead pointer: ≤ +4 indices and within 0.18 m. PP env: pointer jumps to the nearest index in a [−3,+21] window (mod-N rule on loops) | Two different metric definitions in one column | Ma | Single post-hoc sequential-coverage function applied to every trajectory | `metrics` | Toy-trajectory unit tests | unresolved |
| 7 | "each learned controller is a single trained model" (disclosed) | — | No training-seed variance | Ma | ≥5 independent training seeds per learned controller | experiments | Per-seed records | unresolved |
| 8 | Fixed PP (L=0.55, g=1.05), Adaptive PP, RL-PP teacher coefficients | No record of how these were chosen or on which data | Baseline tuning protocol unknown; possible tuning on evaluation circuits | Ma | Grid-search Fixed PP on training circuits only; report both original and tuned values | `rl_pure_pursuit_env.py`, §III-F | Tuning log | unresolved |
| 9 | Seeded protocol | `BeadEnvironment.__init__` and `RLPurePursuitEnv` call `np.random.seed` (global); `BeadTraceEnv.reset(seed=…)` ignores the seed; DAgger β coin-flips use the global stream that every env construction resets | Hidden coupling of unrelated randomness; gym seeding API not honoured | Mo | Per-instance `np.random.Generator`; honour `reset(seed=)` | `environment.py`, `bead_gym_env.py`, `rl_pure_pursuit_env.py` | Determinism test | unresolved |
| 10 | Image → reference | Unreadable image or no extractable path silently falls back to a 4 m circle (print only) | An evaluation can silently score a circle | Mo | `strict` mode that raises; failures recorded | `environment.py` | Unit test | unresolved |
| 11 | "Episodes are truncated after 3500 steps" | 3500 steps × 0.016 s at v* ≈ 1.27 m/s ≈ 71 m; circuits reach 82 m. PP env uses 2000 steps | Cap prevents completion on long circuits and differs between systems | Mo | Path-length-dependent cap, identical for every controller on a given path | `environment.py`, `rl_pure_pursuit_env.py` | Expert completes every circuit | unresolved |
| 12 | Metrics measured against "the reference" | Reference itself deviates 1 cm (mean) / ≤6.5 cm (max) from the source centerline | Perception error is the same order as the tracking RMSE; not reported | Mo | Report extraction error separately | §IV, new table | Ground-truth comparison script | unresolved |
| 13 | Reward Eq. (3) "defines the task" | Never used for learning; per-step centring bonus (0.25) dominates the return (∝ episode length); tangency term divided by v_max = 7 m/s although the bead cannot exceed 1.79 m/s, so its effective weight is ≤ 0.038 | Untested reward; possible incentive to dawdle | Mo | Log reward components during PPO; check speed/progress for exploitation; change weights only with evidence | `environment.py` | Component logs; traversal-time comparison | unresolved |
| 14 | Observation design | Includes absolute arena position and normalised progress index | Lets a policy memorise circuit-specific maps; harmful when train = test | Mo | Keep (preserve design) but evaluate on held-out circuits; ablation if budget allows | `environment.py` | Held-out evaluation | unresolved |
| 15 | Table IV "zero-shot held-out" | Synthetic test split is i.i.d. from the same generator as train/val (same shape families) | Not a geometry-family hold-out | Mi | State it as in-distribution synthetic test | §IV-A | — | unresolved |
| 16 | "coverage entries vary by less than 0.3 pp" | Range up to 0.86 pp | Incorrect statement | Mi | Remove/replace with measured values | §V-A | stored records | verified incorrect |
| 17 | Paper text | Placeholders: author block "[Department]…", Table IV footnote "[standard deviation across the 25 shapes?]", "[gmin], [gmax]", "[Lmin], [Lmax]" | Unfinished manuscript | Mi | Fill or remove | paper | — | unresolved |
| 18 | README | Unresolved git merge-conflict markers; describes `COMPLETE_BTP_MODEL.zip` as the BTP model although it is the RL-PP checkpoint (`train_rl_pure_pursuit.py` mirrors saves to that name); stray file `h -u origin main --force-with-lease`; README "Paper" columns mix [4]'s hardware lap times with this repo's simulation | Misleading documentation | Mi | Rewrite README | `README.md` | — | unresolved |
| 19 | Reproducibility | `requirements.txt` unpinned; RL-PP trainer writes JSON/zip files to the CWD | | Mi | Pin versions; explicit output dirs | `requirements.txt`, trainer | — | unresolved |
| 20 | Eq. (1)–(2) dynamics, Table I weights, Eq. (3) structure, 16-D obs, Eq. (4) expert, §III-A pipeline, Alg. 1 schedule/learning rates, Table II RL-PP hyper-parameters | Match the code | — | — | none | — | code inspection | verified |
| 21 | References [1]–[22] | [4] arXiv:2602.18386, [5] 1707.06347, [22] 2407.17032 confirmed online; others are standard, well-known works whose bibliographic details are consistent with the published versions as far as I can verify without library access | — | — | none | — | arXiv lookups | [4],[5],[22] verified; rest not independently fetched |
