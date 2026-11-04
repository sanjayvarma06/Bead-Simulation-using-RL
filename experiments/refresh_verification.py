"""Refreshes results/final/verification/report.json after controllers are added to episodes.csv:
accounting and record checks over all rows (verify_results), and re-execution of the new controllers
from their stored configuration; earlier re-execution results are kept.

Usage: python experiments/refresh_verification.py [name1,name2,...]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_SEEDS, split_ids  # noqa: E402
from valeval import Evaluator  # noqa: E402
from evaluate import controllers  # noqa: E402
import verify_results as V  # noqa: E402


def main():
    new = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set()
    path = V.OUT / "report.json"
    rep = json.loads(path.read_text())
    df = pd.read_csv(V.FINAL / "episodes.csv")
    rep["A_accounting"] = V.check_accounting(df)
    rep["B_C_records"] = V.check_records(df)
    if new:
        ev = Evaluator(4)
        ids = split_ids("test", "f1tenth_circuit")
        for c in controllers():
            if c["name"] not in new:
                continue
            res = ev.run(c["kind"], c["model"], ids, EVAL_SEEDS, **dict(c.get("kw", {}), keep_traj=True))
            T = np.load(V.FINAL / "traj" / f"{c['name']}__s{c['seed']}.npz")
            diffs, same = [], True
            for m, ex in res:
                old, nw = T[f"{m['id']}__{m['seed']}"], ex["traj"].astype(np.float32)
                same = same and len(old) == len(nw)
                diffs.append(float(np.abs(old - nw).max()) if len(old) == len(nw) else float("inf"))
            rep["D_rerun"][f"{c['name']}/seed{c['seed']}"] = {"episodes": len(res), "same_length": same,
                                                              "max_abs_position_diff_m": max(diffs)}
        ev.close()
    path.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep["A_accounting"].items() if k != "references_by_split_family"}, indent=1))
    print("missing", len(rep["B_C_records"]["missing_trajectories"]), "rows", rep["B_C_records"]["rows_checked"],
          "indep", rep["B_C_records"]["max_abs_diff_independent_code_vs_stored"], "rerun instances", len(rep["D_rerun"]))
    print({k: v for k, v in rep["D_rerun"].items() if k.split("/")[0] in new})


if __name__ == "__main__":
    main()
