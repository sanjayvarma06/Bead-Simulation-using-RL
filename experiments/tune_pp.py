"""Grid-search tuning of the geometric PP baselines on TRAINING references only.

Fixed PP (car):     (L_d, g) over L_GRID x G_GRID.
Adaptive PP (car):  L_d = clip(a + b v, L_MIN, 1.20), g = 1.05, (a, b) over A_GRID x B_GRID.
Bead-PP:            the Fixed PP law driven on the bead plant at v_match (common.BeadPursuit),
                    over the same L_GRID x G_GRID as the car.
The car baselines are tuned at the native speed profile and at the matched speed (v_cap = v_match
from runs/expert_trainval.json, derived from the expert on training circuits).
The selection score is the same `val_score` used for every learned controller, computed on the
training subset; only configurations that complete every tuning reference are eligible. The
selected configuration is then scored once on validation. Whether a selected value lies on the
edge of its grid is recorded, so that edge optima can be detected and the grid widened.

Usage: python experiments/tune_pp.py [workers]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402
from rl_pure_pursuit_env import L_MIN  # noqa: E402

r = lambda v: round(float(v), 3)  # noqa: E731
L_GRID = [r(x) for x in np.arange(L_MIN, 0.2001, 0.025)] + [r(x) for x in np.arange(0.25, 1.001, 0.05)]
G_GRID = [r(x) for x in np.arange(0.6, 1.4001, 0.1)]
A_GRID = [r(x) for x in np.arange(0.0, 0.5001, 0.025)]
B_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25]
SEED = [777]


def tuning_ids():
    """All 11 training circuits + the first 15 training shapes of each synthetic family."""
    return (split_ids("train", "f1tenth_circuit") + split_ids("train", "closed_shape")[:15]
            + split_ids("train", "open_curve")[:15])


def kwargs_for(kind, p, v_cap):
    if kind == "fixed":
        return "pp", {"mode": "fixed", "v_cap": v_cap, "fixed_L": p[0], "fixed_g": p[1]}
    if kind == "adaptive":
        return "pp", {"mode": "adaptive", "v_cap": v_cap, "adapt_a": p[0], "adapt_b": p[1], "fixed_g": 1.05}
    return "bead_geo", {"ctrl": "pp", "L_d": p[0], "g": p[1], "v_ref": v_cap}


def search(ev, kind, grid, v_cap):
    rows = []
    ids = tuning_ids()
    for p in grid:
        ev_kind, kw = kwargs_for(kind, p, v_cap)
        res = [m for m, _ in ev.run(ev_kind, None, ids, SEED, **kw)]
        rows.append({"params": list(p), "completed_all": float(np.mean([m["completed"] for m in res])),
                     **val_score(res)})
        print(kind, v_cap, p, round(rows[-1]["score"], 4), round(rows[-1]["completed_all"], 3), flush=True)
    eligible = [x for x in rows if x["completed_all"] == 1.0] or rows
    return max(eligible, key=lambda x: x["score"]), rows


def on_edge(best, axes):
    return [bool(v in (ax[0], ax[-1])) for v, ax in zip(best["params"], axes)]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    v_match = json.loads((RUNS / "expert_trainval.json").read_text())["v_match"]
    ev = Evaluator(workers)
    out = {"v_match": v_match, "tuning_ids": tuning_ids(), "L_grid": L_GRID, "g_grid": G_GRID,
           "a_grid": A_GRID, "b_grid": B_GRID}
    fg = [(L, g) for L in L_GRID for g in G_GRID]
    ag = [(a, b) for a in A_GRID for b in B_GRID]
    jobs = [("fixed", "native", None, fg, (L_GRID, G_GRID)), ("adaptive", "native", None, ag, (A_GRID, B_GRID)),
            ("fixed", "matched", v_match, fg, (L_GRID, G_GRID)), ("adaptive", "matched", v_match, ag, (A_GRID, B_GRID)),
            ("beadpp", "matched", v_match, fg, (L_GRID, G_GRID))]
    for kind, tag, v_cap, grid, axes in jobs:
        best, rows = search(ev, kind, grid, v_cap)
        ev_kind, kw = kwargs_for(kind, best["params"], v_cap)
        val = val_score([m for m, _ in ev.run(ev_kind, None, split_ids("val"), SEED, **kw)])
        out[f"{kind}_{tag}"] = {"best": best, "val": val, "kwargs": kw, "on_edge": on_edge(best, axes), "grid": rows}
        print("SELECTED", kind, tag, best["params"], "edge", on_edge(best, axes), "val", round(val["score"], 4), flush=True)
        (RUNS / "pp_tuning.json").write_text(json.dumps(out, indent=1))
    ev.close()


if __name__ == "__main__":
    main()
