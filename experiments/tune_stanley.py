"""Grid-search tuning of the Stanley baseline on TRAINING references only.

Stanley [Hoffmann et al. 2007] on the same kinematic car as PP:
    delta = (path heading - heading) + atan(k * e_front / (k_s + v)),
with gains (k, k_s) over K_GRID x KS_GRID, at the native speed profile and at v_match.
Identical protocol to tune_pp.py (same tuning references, same selection score `val_score`,
only configurations completing every tuning reference are eligible, one validation run of the
selected configuration, edge detection).

Usage: python experiments/tune_stanley.py [workers]
Output: runs/stanley_tuning.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402
import tune_pp  # noqa: E402

# First grid: k 0.25-24, k_s 0.05-4. Its optimum lay on the edge (k = 24, k_s = 0.05) at both speeds,
# so the grid was widened (k up to 128, k_s down to 0.01). The optimum stayed on the edge at k = 512,
# so it was widened again until the optimum is interior; evaluated configurations are cached.
K_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]
KS_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def kwargs_for(kind, p, v_cap):
    return "pp", {"mode": "stanley", "v_cap": v_cap, "stanley_k": p[0], "stanley_ks": p[1]}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    v_match = json.loads((RUNS / "expert_trainval.json").read_text())["v_match"]
    tune_pp.kwargs_for = kwargs_for          # reuse tune_pp.search with the Stanley parameterisation
    old = json.loads((RUNS / "stanley_tuning.json").read_text()) if (RUNS / "stanley_tuning.json").exists() else {}
    ev = Evaluator(workers)
    out = {"v_match": v_match, "tuning_ids": tune_pp.tuning_ids(), "k_grid": K_GRID, "ks_grid": KS_GRID}
    grid = [(k, ks) for k in K_GRID for ks in KS_GRID]
    for tag, v_cap in (("native", None), ("matched", v_match)):
        done = {tuple(r["params"]): r for r in old.get(f"stanley_{tag}", {}).get("grid", [])}
        _, rows_new = tune_pp.search(ev, "stanley", [g for g in grid if tuple(g) not in done], v_cap)
        rows = list(done.values()) + rows_new
        eligible = [x for x in rows if x["completed_all"] == 1.0] or rows
        best = max(eligible, key=lambda x: x["score"])
        _, kw = kwargs_for("stanley", best["params"], v_cap)
        val = val_score([m for m, _ in ev.run("pp", None, split_ids("val"), tune_pp.SEED, **kw)])
        out[f"stanley_{tag}"] = {"best": best, "val": val, "kwargs": kw,
                                 "on_edge": tune_pp.on_edge(best, (K_GRID, KS_GRID)), "grid": rows}
        print("SELECTED stanley", tag, best["params"], "edge", out[f"stanley_{tag}"]["on_edge"],
              "val", round(val["score"], 4), flush=True)
        (RUNS / "stanley_tuning.json").write_text(json.dumps(out, indent=1))
    ev.close()


if __name__ == "__main__":
    main()
