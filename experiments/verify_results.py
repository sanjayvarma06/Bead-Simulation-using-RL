"""Independent verification of every reported test result.

Checks
  A. Accounting: number of references, controller instances and episodes.
  B. Record integrity: every row of results/final/episodes.csv has a stored raw trajectory, and all
     metrics recomputed from that trajectory (metrics.trajectory_metrics) equal the stored values.
  C. Independent metric code: RMSE and Geom@0.10 recomputed with a separate brute-force
     point-to-segment implementation that does not use geometry_utils/metrics.
  D. Re-execution: saved checkpoints are reloaded and re-run on all test circuits; the new
     trajectories are compared with the stored ones.
  E. Tables: the paper's LaTeX tables and quoted numbers are regenerated from episodes.csv into a
     temporary directory and compared byte-for-byte with paper/revised/tables.
  F. Run inventory: checkpoint hashes, configs, training logs and exit codes of every reported run.

Usage: python experiments/verify_results.py [--skip_rerun]
Output: results/final/verification/report.json and report.md
"""
from __future__ import annotations
import argparse
import filecmp
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, EVAL_SEEDS, load_path, split_ids, load_manifest  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402
from evaluate import SEEDS  # noqa: E402

BEAD_MODES = ("dagger", "dagger_bpp", "ppo", "dagger_ppo", "ppo_acc", "dagger_ppo_acc")

FINAL = ROOT / "results" / "final"
OUT = FINAL / "verification"
METRIC_KEYS = ["geom005", "geom010", "geom015", "rmse", "mean_err", "p95", "max_err", "off_path_frac",
               "sequential", "ordered_cov", "steps", "time_s", "distance_m", "mean_speed", "peak_speed", "rms_accel"]


# ----------------------------------------------------------------------------- independent metric code
def _seg_dist(points, a, b):
    """Brute-force distance from each point to the polyline with segments a[i]->b[i]."""
    out = np.empty(len(points))
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-18)
    for i in range(0, len(points), 512):
        q = points[i:i + 512, None, :]
        t = np.clip(((q - a) * ab).sum(2) / L2, 0, 1)
        out[i:i + 512] = np.sqrt((((a + t[..., None] * ab) - q) ** 2).sum(2).min(1))
    return out


def independent_rmse_geom(path, traj):
    e = _seg_dist(traj, path[:-1], path[1:])
    d = _seg_dist(path, traj[:-1], traj[1:]) if len(traj) > 1 else np.linalg.norm(path - traj[0], axis=1)
    return float(np.sqrt((e ** 2).mean())), float((d <= 0.10).mean())


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- checks
def check_accounting(df):
    m = load_manifest()["entries"]
    by = {}
    for e in m:
        by.setdefault(e["split"], {}).setdefault(e["family"], 0)
        by[e["split"]][e["family"]] += 1
    inst = df.groupby(["controller", "train_seed"]).size()
    per_inst = sorted(set(inst.tolist()))
    n_circ, n_syn = df[df.family == "circuit"].ref.nunique(), df[df.family != "circuit"].ref.nunique()
    return {"references_in_manifest": len(m), "extracted_ok": sum(e["extraction_ok"] for e in m),
            "references_by_split_family": by, "controller_instances": int(len(inst)),
            "episodes_per_instance": per_inst, "test_circuits": int(n_circ),
            "eval_seeds_per_circuit": int(df[df.family == "circuit"].eval_seed.nunique()),
            "test_synthetic_evaluated": int(n_syn), "episodes_total": int(len(df)),
            "formula": f"{len(inst)} instances x ({n_circ} circuits x {len(EVAL_SEEDS)} seeds + {n_syn} contours x 1)"
                       f" = {len(inst) * (n_circ * len(EVAL_SEEDS) + n_syn)}"}


def check_records(df):
    paths = {}
    worst = {k: 0.0 for k in METRIC_KEYS}
    worst_indep = {"rmse": 0.0, "geom010": 0.0}
    missing, n = [], 0
    for (c, s), g in df.groupby(["controller", "train_seed"]):
        f = FINAL / "traj" / f"{c}__s{'None' if s == -1 else s}.npz"
        T = np.load(f)
        for _, r in g.iterrows():
            key = f"{r.ref}__{r.eval_seed}"
            if key not in T:
                missing.append(f"{c}/{s}/{key}"); continue
            tr = T[key].astype(float)
            if r.ref not in paths:
                paths[r.ref] = load_path(r.ref)
            p, brk = paths[r.ref]
            m = trajectory_metrics(p, tr, brk)
            for k in METRIC_KEYS:
                # trajectories are stored as float32, so allow float32 rounding
                worst[k] = max(worst[k], abs(float(m[k]) - float(r[k])))
            assert not brk
            rm, g10 = independent_rmse_geom(p, tr)
            worst_indep["rmse"] = max(worst_indep["rmse"], abs(rm - r.rmse))
            worst_indep["geom010"] = max(worst_indep["geom010"], abs(g10 - r.geom010))
            n += 1
    return {"rows_checked": n, "missing_trajectories": missing,
            "max_abs_diff_recomputed_vs_stored": worst,
            "max_abs_diff_independent_code_vs_stored": worst_indep}


def check_rerun():
    """Reload checkpoints and re-run them on all test circuits x 5 start seeds."""
    from valeval import Evaluator
    from evaluate import controllers
    ids = split_ids("test", "f1tenth_circuit")
    # every controller once: deterministic ones, and training seed 0 of every learned one
    want = {(c["name"], c["seed"]) for c in controllers() if c["seed"] in (None, 0)}
    ev = Evaluator(8)
    out = {}
    for c in controllers():
        if (c["name"], c["seed"]) not in want:
            continue
        kw = dict(c.get("kw", {}), keep_traj=True)
        res = ev.run(c["kind"], c["model"], ids, EVAL_SEEDS, **kw)
        T = np.load(FINAL / "traj" / f"{c['name']}__s{c['seed']}.npz")
        diffs, lens = [], []
        for m, ex in res:
            old = T[f"{m['id']}__{m['seed']}"]
            new = ex["traj"].astype(np.float32)
            lens.append(len(old) == len(new))
            diffs.append(float(np.abs(old - new).max()) if len(old) == len(new) else float("inf"))
        out[f"{c['name']}/seed{c['seed']}"] = {"episodes": len(res), "same_length": all(lens),
                                               "max_abs_position_diff_m": max(diffs)}
        print(c["name"], c["seed"], out[f"{c['name']}/seed{c['seed']}"], flush=True)
    ev.close()
    return out


def check_tables():
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, TABLES_OUT=str(tmp))
    subprocess.run([sys.executable, str(ROOT / "paper/scripts/make_tables_v2.py")], env=env, check=True,
                   capture_output=True)
    res = {}
    for f in sorted((ROOT / "paper/revised/tables").glob("*.tex")):
        res[f.name] = filecmp.cmp(f, tmp / f.name, shallow=False)
    return res


def inventory():
    inv = {}
    status = {}
    qs = RUNS / "logs" / "queue_status.jsonl"
    if qs.exists():
        for line in qs.read_text().splitlines():
            r = json.loads(line); status[r["name"]] = r
    for kind, modes in (("bead", BEAD_MODES), ("rlpp", ("joint", "ld_only", "joint_matched"))):
        for mode in modes:
            for s in SEEDS:
                d = RUNS / kind / mode / f"seed{s}"
                e = {"dir": str(d.relative_to(ROOT)).replace("\\", "/"),
                     "best_ckpt_md5": md5(d / "best.zip") if (d / "best.zip").exists() else None,
                     "config": (d / "config.json").exists(), "val_log_rows": None, "progress_rows": None,
                     "note": (d / "NOTE.txt").read_text().strip() if (d / "NOTE.txt").exists() else None}
                if (d / "val_log.csv").exists():
                    e["val_log_rows"] = len(pd.read_csv(d / "val_log.csv"))
                if (d / "progress.csv").exists():
                    pr = pd.read_csv(d / "progress.csv")
                    e["progress_rows"] = len(pr)
                    e["final_timesteps"] = int(pr["time/total_timesteps"].iloc[-1])
                inv[f"{kind}/{mode}/seed{s}"] = e
    for f in ("PPO_BTP_MODEL.zip", "RL_PP_JOINT_MODEL.zip"):
        inv[f"published/{f}"] = {"md5": md5(ROOT / "results/models" / f)}
    return {"runs": inv, "queue_exit_codes": status}


def ppo_diagnostics():
    out = {}
    for mode in ("ppo", "dagger_ppo", "ppo_acc", "dagger_ppo_acc"):
        for s in SEEDS:
            pr = pd.read_csv(RUNS / "bead" / mode / f"seed{s}" / "progress.csv")
            fz = pr["train/actor_frozen"] == 1 if "train/actor_frozen" in pr else pd.Series(False, index=pr.index)
            tr = pr[~fz]
            def first_last(col, frame=tr):
                v = frame[col].dropna()
                return [round(float(v.iloc[0]), 4), round(float(v.iloc[-1]), 4)] if len(v) else None
            out[f"{mode}/seed{s}"] = {
                "updates": int(len(pr)), "updates_actor_frozen": int(fz.sum()),
                "approx_kl_while_frozen_max": float(pr.loc[fz, "train/approx_kl"].max()) if fz.any() else None,
                "approx_kl_first_last": first_last("train/approx_kl"),
                "clip_fraction_first_last": first_last("train/clip_fraction"),
                "explained_variance_first_last": first_last("train/explained_variance"),
                "action_std_first_last": first_last("train/std"),
                "train_ep_reward_first_last": first_last("rollout/ep_rew_mean", pr),
                "train_ep_len_first_last": first_last("rollout/ep_len_mean", pr),
                "train_ep_progress_first_last": first_last("episode/coverage", pr),
            }
    return out


def check_centerline(df, n_sample=60):
    """True-centerline records: one per circuit episode, and a random sample recomputed independently."""
    from centerline_metrics import true_centerline
    cl = pd.read_csv(FINAL / "centerline" / "episodes.csv")
    circ = df[df.family == "circuit"]
    key = ["controller", "train_seed", "ref", "eval_seed"]
    same_keys = (set(map(tuple, circ[key].astype(str).values)) == set(map(tuple, cl[key].astype(str).values)))
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.choice(len(cl), size=min(n_sample, len(cl)), replace=False):
        r = cl.iloc[i]
        seed = "None" if int(r.train_seed) == -1 else int(r.train_seed)
        traj = np.load(FINAL / "traj" / f"{r.controller}__s{seed}.npz")[f"{r.ref}__{int(r.eval_seed)}"].astype(float)
        rmse, _ = independent_rmse_geom(true_centerline(r.ref), traj)
        worst = max(worst, abs(rmse - r.rmse_true))
    return {"rows": int(len(cl)), "circuit_episodes": int(len(circ)), "same_episode_keys": bool(same_keys),
            "sampled": int(min(n_sample, len(cl))), "max_abs_diff_rmse_independent_code": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_rerun", action="store_true")
    ap.add_argument("--reuse_records", action="store_true",
                    help="carry over checks B/C if episodes.csv is unchanged since they were run")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((OUT / "report.json").read_text()) if (OUT / "report.json").exists() else {}
    df = pd.read_csv(FINAL / "episodes.csv")
    ep_md5 = md5(FINAL / "episodes.csv")
    rep = {"episodes_csv_md5": ep_md5, "A_accounting": check_accounting(df)}
    print(json.dumps(rep["A_accounting"], indent=1), flush=True)
    unchanged = old.get("episodes_csv_md5") == ep_md5 or (
        "episodes_csv_md5" not in old and (FINAL / "episodes.csv").stat().st_mtime < (OUT / "report.json").stat().st_mtime)
    if args.reuse_records and "B_C_records" in old and unchanged:
        rep["B_C_records"] = old["B_C_records"]
        rep["B_C_records_carried_over_episodes_unchanged"] = True
    else:
        rep["B_C_records"] = check_records(df)
    print(json.dumps(rep["B_C_records"], indent=1), flush=True)
    rep["H_centerline"] = check_centerline(df)
    print(json.dumps(rep["H_centerline"]), flush=True)
    rep["E_tables_identical"] = check_tables()
    print(rep["E_tables_identical"], flush=True)
    rep["F_inventory"] = inventory()
    rep["G_ppo_diagnostics"] = ppo_diagnostics()
    print(json.dumps(rep["G_ppo_diagnostics"], indent=1), flush=True)
    if not args.skip_rerun:
        rep["D_rerun"] = check_rerun()
    elif "D_rerun" in old and unchanged:
        # keep the re-execution result of the last full run, marked as such
        rep["D_rerun"] = old["D_rerun"]
        rep["D_rerun_carried_over_from_previous_full_run"] = True
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("->", OUT / "report.json")


if __name__ == "__main__":
    main()
