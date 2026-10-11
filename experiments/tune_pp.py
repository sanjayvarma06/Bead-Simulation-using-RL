"""Grid-search tuning of the classical PP baselines on TRAINING references only.

Fixed PP:    (L_d, g) over a 16 x 6 grid.
Adaptive PP: L_d = clip(a + b v), g = 1.05, (a, b) over a 9 x 6 grid.
Both at the native speed profile and at the matched speed (v_cap = v_match from
runs/expert_trainval.json, derived from the expert on training circuits).
The selection score is the same `val_score` used for every learned controller, computed
on the training subset; the selected configuration is then scored once on validation.

Usage: python experiments/tune_pp.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402

FIXED_GRID = [(round(L, 2), round(g, 2)) for L in np.arange(0.25, 1.01, 0.05) for g in np.arange(0.8, 1.31, 0.1)]
ADAPT_GRID = [(round(a, 2), round(b, 2)) for a in np.arange(0.10, 0.51, 0.05) for b in np.arange(0.0, 0.26, 0.05)]
SEED = [777]


def tuning_ids():
    """All 11 training circuits + the first 15 training shapes of each synthetic family."""
    return (split_ids("train", "f1tenth_circuit") + split_ids("train", "closed_shape")[:15]
            + split_ids("train", "open_curve")[:15])


def search(ev, mode, grid, v_cap):
    rows = []
    ids = tuning_ids()
    for p in grid:
        kw = {"mode": mode, "v_cap": v_cap}
        if mode == "fixed":
            kw.update(fixed_L=p[0], fixed_g=p[1])
        else:
            kw.update(adapt_a=p[0], adapt_b=p[1], fixed_g=1.05)
        res = [r for r, _ in ev.run("pp", None, ids, SEED, **kw)]
        s = val_score(res)
        comp = float(np.mean([r["completed"] for r in res]))
        rows.append({"params": p, "completed_all": comp, **s})
        print(mode, v_cap, p, round(s["score"], 4), round(comp, 3), flush=True)
    # Only configurations that complete every tuning reference are eligible (fallback: all).
    eligible = [r for r in rows if r["completed_all"] == 1.0] or rows
    best = max(eligible, key=lambda r: r["score"])
    return best, rows


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    v_match = json.loads((RUNS / "expert_trainval.json").read_text())["v_match"]
    ev = Evaluator(workers)
    out = {"v_match": v_match, "tuning_ids": tuning_ids()}
    for tag, v_cap in (("native", None), ("matched", v_match)):
        for mode, grid in (("fixed", FIXED_GRID), ("adaptive", ADAPT_GRID)):
            best, rows = search(ev, mode, grid, v_cap)
            kw = {"mode": mode, "v_cap": v_cap}
            if mode == "fixed":
                kw.update(fixed_L=best["params"][0], fixed_g=best["params"][1])
            else:
                kw.update(adapt_a=best["params"][0], adapt_b=best["params"][1], fixed_g=1.05)
            val = val_score([r for r, _ in ev.run("pp", None, split_ids("val"), SEED, **kw)])
            out[f"{mode}_{tag}"] = {"best": best, "val": val, "kwargs": kw, "grid": rows}
            print("SELECTED", mode, tag, best["params"], "val", round(val["score"], 4), flush=True)
    ev.close()
    (RUNS / "pp_tuning.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
