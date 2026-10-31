"""Robustness check: tuned Fixed PP and RL-PP at matched speed on a DYNAMIC single-track car.

Two variants: "dynamic_norate" (linear-tyre single-track model only) and "dynamic" (the same with the
F1TENTH Gym steering-rate limit of 3.2 rad/s, which is a full-scale actuator value).

The controllers are taken unchanged from the kinematic study (Fixed PP gains tuned at v_match on
the kinematic car; RL-PP policies trained at v_match on the kinematic car, five seeds, with their
VecNormalize statistics) and run on the F1TENTH Gym single-track model with linear tyres
(research_paper_code/dynamic_single_track.py) at the same capped speed v_match, on the eight test
circuits with the same five start seeds. No retuning or retraining is done.

Usage: python experiments/dynamic_check.py [workers]
Output: results/final/dynamic/episodes.csv, summary.csv, comparisons.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, EVAL_SEEDS, split_ids  # noqa: E402
from valeval import Evaluator  # noqa: E402

FINAL = ROOT / "results" / "final"
OUT = FINAL / "dynamic"
NAMES = ("pp_fixed_tuned_matched", "rlpp_joint_matched")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    OUT.mkdir(parents=True, exist_ok=True)
    ctrls = [c for c in json.loads((FINAL / "controllers.json").read_text()) if c["name"] in NAMES]
    circ = split_ids("test", "f1tenth_circuit")
    ev = Evaluator(workers)
    rows = []
    for c, variant in [(c, v) for v in ("dynamic_norate", "dynamic") for c in ctrls]:
        kw = dict(c.get("kw", {}), dynamics=variant)
        if kw.get("vecnorm"):
            kw["vecnorm"] = str(ROOT / kw["vecnorm"])
        model = None if c["model"] in (None, "expert") else str(ROOT / c["model"])
        for m, _ in ev.run("pp", model, circ, EVAL_SEEDS, **kw):
            rows.append({"model": variant, "controller": c["name"], "train_seed": -1 if c["seed"] is None else c["seed"],
                         "ref": m["id"], "eval_seed": m["seed"], "rmse": m["rmse"], "p95": m["p95"],
                         "max_err": m["max_err"], "geom010": m["geom010"], "completed": bool(m["completed"]),
                         "time_s": m["time_s"], "mean_speed": m["mean_speed"]})
        print(variant, c["name"], c["seed"], "done", flush=True)
    ev.close()
    dyn = pd.DataFrame(rows)
    dyn.to_csv(OUT / "episodes.csv", index=False)
    analyse(dyn)


def analyse(dyn: pd.DataFrame):
    from analyze import bootstrap_ci, wilcoxon_exact
    kin = pd.read_csv(FINAL / "episodes.csv")
    kin = kin[(kin.family == "circuit") & kin.controller.isin(NAMES)]
    kin["completed"] = kin["completed"].astype(str).str.lower().eq("true")
    both = pd.concat([kin.assign(model="kinematic"), dyn], ignore_index=True)
    both["completed"] = both["completed"].astype(float)
    prs = (both.groupby(["model", "controller", "train_seed", "ref"])[["rmse", "p95", "geom010", "completed", "time_s"]]
           .mean().reset_index())
    summ = (prs.groupby(["model", "controller", "train_seed"]).mean(numeric_only=True)
            .groupby(["model", "controller"]).agg(["mean", "min", "max"]))
    summ.to_csv(OUT / "summary.csv")
    # paired comparisons over the eight circuits (bootstrap over circuits and seeds)
    prs["family"] = "circuit"
    lab = prs.copy()
    lab["controller"] = lab["model"] + ":" + lab["controller"]
    out = []
    F, RL = "pp_fixed_tuned_matched", "rlpp_joint_matched"
    pairs = []
    for v in ("dynamic_norate", "dynamic"):
        pairs += [(f"{v}:{F}", f"kinematic:{F}"), (f"{v}:{RL}", f"kinematic:{RL}"), (f"{v}:{RL}", f"{v}:{F}")]
    for a, b in pairs:
        point, lo, hi, per_ref = bootstrap_ci(lab, a, b, "rmse")
        p, _ = wilcoxon_exact(per_ref)
        out.append({"A": a, "B": b, "mean_diff_A_minus_B": point, "ci95_lo": lo, "ci95_hi": hi,
                    "A_better_on": int((per_ref < 0).sum()), "n_refs": len(per_ref), "wilcoxon_p": p})
    pd.DataFrame(out).to_csv(OUT / "comparisons.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print((100 * summ.xs("rmse", axis=1, level=0)).round(2))
        print(pd.DataFrame(out).round(5))


if __name__ == "__main__":
    analyse(pd.read_csv(OUT / "episodes.csv")) if "--analyse" in sys.argv else main()
