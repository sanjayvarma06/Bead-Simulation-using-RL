"""Robustness experiments on the 8 test circuits (5 start seeds each); every number from rollouts.

  dynamic : F1TENTH single-track car at v_match, tyres only ("dt") and with the 3.2 rad/s
            steering-rate limit ("dr"). Geometric laws re-tuned on that model (tune_extra.py),
            RL-PP retrained on it (train_rlpp.py --dynamics), and the kinematically tuned / trained
            controllers run unchanged ("transfer").
  real    : true-scale circuits, F1TENTH dynamic car with rate limit and 0.4189 rad steering limit,
            v_match; expert and Bead-PP on the bead at the same scale (all re-tuned at that scale).
  fast    : faster bead (a_max = 25 m/s^2) at the car's native pace: expert, Bead-PP following the
            car's speed profile (re-tuned), Bead-DAgger zero-shot.
  noise   : Gaussian noise (sigma = 2 and 5 cm) on the position every controller sees.
  dense   : 1000-waypoint re-extraction of the test circuits; unchanged controllers.
Every trajectory is also scored against the true centerline (arena coordinates, or the reference
itself for the true-scale circuits).

Usage: python experiments/robustness_eval.py <exp> [<exp> ...] [--workers N]
Output: results/final/robustness/<exp>.csv, results/final/robustness/traj/<exp>__<name>__s<seed>.npz
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, EVAL_SEEDS, split_ids, load_path  # noqa: E402
from valeval import Evaluator  # noqa: E402

FINAL = ROOT / "results" / "final"
OUT = FINAL / "robustness"
XT = RUNS / "extra_tuning"
SEEDS = [0, 1, 2, 3, 4]


def base_controllers():
    """Main-study controllers by name -> list of (seed, kind, model, kw)."""
    C = {}
    for c in json.loads((FINAL / "controllers.json").read_text()):
        kw = dict(c.get("kw", {}))
        if kw.get("vecnorm"):
            kw["vecnorm"] = str(ROOT / kw["vecnorm"])
        model = c["model"] if c["model"] in (None, "expert") else str(ROOT / c["model"])
        C.setdefault(c["name"], []).append((c["seed"], c["kind"], model, kw))
    return C


def tuned(job):
    j = json.loads((XT / f"{job}.json").read_text())
    return [(None, j["kind"], None, dict(j["kwargs"]))]


def rlpp_dyn(tag, dyn):
    out = []
    for s in SEEDS:
        d = RUNS / "rlpp" / f"joint_matched_{dyn}" / f"seed{s}"
        out.append((s, "pp", str(d / "best.zip"), {"mode": "joint", "v_cap": json.loads(
            (RUNS / "expert_trainval.json").read_text())["v_match"], "dynamics": dyn, "vecnorm": str(d / "best_vecnorm.pkl")}))
    return out


def with_kw(entries, **extra):
    return [(s, k, m, dict(kw, **extra)) for s, k, m, kw in entries]


def experiments():
    B = base_controllers()
    E = {}
    for tag, dyn in (("dt", "dynamic_norate"), ("dr", "dynamic")):
        E[f"dynamic_{tag}"] = {
            "fixed_retuned": tuned(f"fixed_{tag}"), "fixed_retuned_highg": tuned(f"fixedhigh_{tag}"),
            "stanley_retuned": tuned(f"stanley_{tag}"),
            "rwf_retuned": tuned(f"rwf_{tag}"), "rlpp_retrained": rlpp_dyn(tag, dyn),
            "fixed_transfer": with_kw(B["pp_fixed_tuned_matched"], dynamics=dyn),
            "stanley_transfer": with_kw(B["stanley_tuned_matched"], dynamics=dyn),
            "rlpp_transfer": with_kw(B["rlpp_joint_matched"], dynamics=dyn)}
    E["real"] = {"expert": with_kw(B["expert"], path_set="real", arena=False),
                 "bead_pp": tuned("beadpp_real"), "fixed": tuned("fixed_real"),
                 "stanley": tuned("stanley_real"), "rwf": tuned("rwf_real")}
    E["fast"] = {"expert": with_kw(B["expert"], accel=25.0), "bead_pp": tuned("beadpp_fast"),
                 "bead_dagger": with_kw(B["bead_dagger"], accel=25.0)}
    noisy = ["expert", "bead_pp", "bead_dagger", "bead_dagger_bpp", "pp_fixed_tuned", "pp_fixed_tuned_matched",
             "stanley_tuned_matched", "rwf_tuned_matched", "rlpp_joint"]
    for sig in (0.02, 0.05):
        E[f"noise{int(sig * 100)}"] = {n: with_kw(B[n], pos_noise=sig) for n in noisy if n in B}
    E["dense"] = {n: with_kw(B[n], path_set="dense") for n in
                  ("expert", "bead_pp", "bead_dagger", "bead_dagger_bpp", "pp_fixed_tuned", "pp_fixed_tuned_matched")}
    return E


def true_reference(ref, path_set):
    if path_set == "real":
        return load_path(ref, "real")[0]
    from centerline_metrics import true_centerline
    return true_centerline(ref)


def run(exp, ev):
    from metrics import trajectory_metrics
    E = experiments()[exp]
    circ = split_ids("test", "f1tenth_circuit")
    (OUT / "traj").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, entries in E.items():
        for seed, kind, model, kw in entries:
            res = ev.run(kind, model, circ, EVAL_SEEDS, **dict(kw, keep_traj=True))
            trajs = {}
            for m, extra in res:
                tr = extra["traj"]
                t = trajectory_metrics(true_reference(m["id"], kw.get("path_set")), tr)
                rows.append({"experiment": exp, "controller": name, "train_seed": -1 if seed is None else seed,
                             "ref": m["id"], "eval_seed": m["seed"], "rmse": m["rmse"], "p95": m["p95"],
                             "max_err": m["max_err"], "geom010": m["geom010"], "completed": bool(m["completed"]),
                             "time_s": m["time_s"], "mean_speed": m["mean_speed"], "rmse_true": t["rmse"]})
                trajs[f"{m['id']}__{m['seed']}"] = tr.astype(np.float32)
            np.savez_compressed(OUT / "traj" / f"{exp}__{name}__s{seed}.npz", **trajs)
            print(exp, name, seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{exp}.csv", index=False)
    summ = summarise(df)
    print(summ.to_string())


def summarise(df):
    df = df.copy()
    df["completed"] = df["completed"].astype(str).str.lower().eq("true").astype(float)
    per = df.groupby(["controller", "train_seed", "ref"]).mean(numeric_only=True).groupby(
        ["controller", "train_seed"]).mean(numeric_only=True)
    g = per.groupby("controller")
    out = pd.DataFrame({"n_seeds": g.size(), "rmse_cm": 100 * g.rmse.mean(), "rmse_lo": 100 * g.rmse.min(),
                        "rmse_hi": 100 * g.rmse.max(), "rmse_true_cm": 100 * g.rmse_true.mean(),
                        "completed_pct": 100 * g.completed.mean(), "time_s": g.time_s.mean()})
    return out.round(3)


def main():
    exps = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    ev = Evaluator(workers)
    for e in exps:
        run(e, ev)
    ev.close()


if __name__ == "__main__":
    main()
