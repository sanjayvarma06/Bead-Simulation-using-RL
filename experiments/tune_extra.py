"""Grid tuning for the robustness experiments, with the protocol of tune_pp.py (training references,
selection score `val_score`, only settings completing every tuning reference are eligible, one
validation run of the selected setting, edge detection).

Jobs (python experiments/tune_extra.py <job> [<job> ...] [--workers N]):
  rwf_native, rwf_matched          rear-wheel feedback on the kinematic car (main tables)
  {fixed,stanley,rwf}_dt           dynamic single-track car, tyres only, v_match, arena scale
  {fixed,stanley,rwf}_dr           same plus the F1TENTH steering-rate limit (3.2 rad/s)
  {fixed,stanley,rwf}_real         TRUE-scale circuits, F1TENTH dynamic model with rate limit and
                                   steering limit 0.4189 rad, v_match
  beadpp_real                      Bead-PP on the true-scale circuits at v_match
  beadpp_fast                      Bead-PP on a faster bead (a_max = 25 m/s^2) following the car's
                                   curvature-based speed profile (the car's native pace)
Output: runs/extra_tuning/<job>.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402
import tune_pp  # noqa: E402
import tune_stanley  # noqa: E402

OUT = RUNS / "extra_tuning"
V_MATCH = json.loads((RUNS / "expert_trainval.json").read_text())["v_match"]
FAST_ACCEL = 25.0
REAL = {"path_set": "real", "arena": False}
REAL_CAR = dict(REAL, dynamics="dynamic", max_steer=0.4189, v_cap=V_MATCH, L_max=4.0)

# widened after the first grid (K_e <= 128, K_psi <= 16) selected K_e = 128 on its edge
G_RWF_KIN = ([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], [0.5, 1, 2, 4, 8, 16, 32, 64])
# Dynamic-model retuning uses the same grids as the kinematic study (tune_pp.py, tune_stanley.py).
G_FIXED_DYN = (tune_pp.L_GRID, tune_pp.G_GRID)
G_STANLEY_DYN = (tune_stanley.K_GRID, tune_stanley.KS_GRID)
G_RWF_DYN = ([1, 4, 16, 64, 128], [1, 4, 16])
G_FIXED_REAL = ([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], [0.8, 1.0, 1.2])
G_STANLEY_REAL = ([0.5, 2, 8, 32, 128], [0.1, 0.5, 2.0, 4.0])
G_RWF_REAL = ([0.25, 1, 4, 16, 64], [0.5, 2, 8, 32])
G_BEADPP_REAL = G_FIXED_REAL
G_BEADPP_FAST = ([0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8], [0.6, 0.8, 1.0, 1.2, 1.4])


def arena_ids():
    return tune_pp.tuning_ids()


def real_ids():
    return split_ids("train", "f1tenth_circuit")


def pp(mode, names):
    return lambda p, base: ("pp", dict(base, mode=mode, **dict(zip(names, p))))


def beadpp(p, base):
    return "bead_geo", dict(base, ctrl="pp", L_d=p[0], g=p[1])


JOBS = {
    "rwf_native": (pp("rwf", ("rwf_ke", "rwf_kpsi")), {"v_cap": None}, G_RWF_KIN, arena_ids, split_ids("val")),
    "rwf_matched": (pp("rwf", ("rwf_ke", "rwf_kpsi")), {"v_cap": V_MATCH}, G_RWF_KIN, arena_ids, split_ids("val")),
    "beadpp_fast": (beadpp, {"v_ref": "profile", "accel": FAST_ACCEL}, G_BEADPP_FAST, arena_ids, split_ids("val")),
    "beadpp_real": (beadpp, dict(REAL, v_ref=V_MATCH), G_BEADPP_REAL, real_ids, split_ids("val", "f1tenth_circuit")),
}
for tag, dyn in (("dt", "dynamic_norate"), ("dr", "dynamic")):
    base = {"v_cap": V_MATCH, "dynamics": dyn}
    JOBS[f"fixed_{tag}"] = (pp("fixed", ("fixed_L", "fixed_g")), base, G_FIXED_DYN, arena_ids, split_ids("val"))
    JOBS[f"stanley_{tag}"] = (pp("stanley", ("stanley_k", "stanley_ks")), base, G_STANLEY_DYN, arena_ids, split_ids("val"))
    JOBS[f"rwf_{tag}"] = (pp("rwf", ("rwf_ke", "rwf_kpsi")), base, G_RWF_DYN, arena_ids, split_ids("val"))
# Gains above the shared grid's upper end (g = 1.4), to test whether that edge matters on the dynamic car
G_FIXED_DYN_HIGH = (tune_pp.L_GRID, [1.5, 1.6, 1.8, 2.0, 2.5])
for tag, dyn in (("dt", "dynamic_norate"), ("dr", "dynamic")):
    JOBS[f"fixedhigh_{tag}"] = (pp("fixed", ("fixed_L", "fixed_g")), {"v_cap": V_MATCH, "dynamics": dyn, "g_max": 6.0},
                                (G_FIXED_DYN_HIGH[0], G_FIXED_DYN_HIGH[1] + ([3.0, 3.5, 4.0, 5.0, 6.0] if tag == "dt" else [])),
                                arena_ids, split_ids("val"))
JOBS["fixed_real"] = (pp("fixed", ("fixed_L", "fixed_g")), REAL_CAR, G_FIXED_REAL, real_ids, split_ids("val", "f1tenth_circuit"))
JOBS["stanley_real"] = (pp("stanley", ("stanley_k", "stanley_ks")), REAL_CAR, G_STANLEY_REAL, real_ids, split_ids("val", "f1tenth_circuit"))
JOBS["rwf_real"] = (pp("rwf", ("rwf_ke", "rwf_kpsi")), REAL_CAR, G_RWF_REAL, real_ids, split_ids("val", "f1tenth_circuit"))


def run_job(ev, name):
    build, base, (ax0, ax1), ids_fn, val_ids = JOBS[name]
    ids = ids_fn()
    rows = []
    prev = OUT / f"{name}.json"            # settings already evaluated (same job, same ids) are reused
    cache = {}
    if prev.exists():
        old = json.loads(prev.read_text())
        if old.get("tuning_ids") == ids:
            cache = {tuple(r["params"]): r for r in old["grid"]}
    for p in [(a, b) for a in ax0 for b in ax1]:
        if tuple(p) in cache:
            rows.append(cache[tuple(p)])
            continue
        kind, kw = build(p, base)
        res = [m for m, _ in ev.run(kind, None, ids, tune_pp.SEED, **kw)]
        rows.append({"params": list(p), "completed_all": float(np.mean([m["completed"] for m in res])), **val_score(res)})
        print(name, p, round(rows[-1]["score"], 4), round(rows[-1]["completed_all"], 3), flush=True)
    eligible = [r for r in rows if r["completed_all"] == 1.0]
    best = max(eligible or rows, key=lambda r: r["score"])
    kind, kw = build(best["params"], base)
    val = val_score([m for m, _ in ev.run(kind, None, val_ids, tune_pp.SEED, **kw)])
    out = {"job": name, "kind": kind, "kwargs": kw, "best": best, "val": val, "any_eligible": bool(eligible),
           "on_edge": tune_pp.on_edge(best, (ax0, ax1)), "grid_axes": [ax0, ax1], "tuning_ids": ids, "grid": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
    print("SELECTED", name, best["params"], "edge", out["on_edge"], "eligible", bool(eligible),
          "train rmse_c", round(100 * best["rmse_circuits"], 3), "val", round(val["score"], 4), flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    args = [a for a in args if a not in (str(workers),)]
    ev = Evaluator(workers)
    for name in args:
        run_job(ev, name)
    ev.close()


if __name__ == "__main__":
    main()
