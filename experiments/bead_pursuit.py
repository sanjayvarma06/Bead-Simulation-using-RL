"""Expert-k sweep: the expert of Eq. (4) aiming k = 1..10 waypoints ahead (bead plant).

Two anchors (common.ExpertK): the waypoint counter w_t (Eq. 4 as defined) and the tracked nearest
reference point. k is selected per anchor on the PP tuning references (training split only) with
the shared score and completion rule; then every k is run once on the test split, as a sensitivity
analysis. Bead-PP (the car's PP law on the bead) is tuned in tune_pp.py and evaluated in evaluate.py.

Outputs: runs/expert_k_tuning.json, results/final/expert_k/{episodes.csv, traj/*.npz}
Usage: python experiments/bead_pursuit.py [workers]
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, EVAL_SEEDS, split_ids  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402
from tune_pp import tuning_ids, SEED  # noqa: E402
from evaluate import N_SYNTH, SYNTH_SEED  # noqa: E402

OUT = ROOT / "results" / "final" / "expert_k"
KS = list(range(1, 11))
ANCHORS = ("counter", "nearest")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ids = tuning_ids()
    ev = Evaluator(workers)
    rows = []
    for anchor in ANCHORS:
        for k in KS:
            kw = {"ctrl": "expert_k", "k": k, "anchor": anchor}
            res = [m for m, _ in ev.run("bead_geo", None, ids, SEED, **kw)]
            rows.append({"kw": kw, "completed_all": float(np.mean([m["completed"] for m in res])), **val_score(res)})
            print("expert-k", anchor, k, round(rows[-1]["score"], 4), rows[-1]["completed_all"], flush=True)
    sel = {}
    for anchor in ANCHORS:
        cand = [r for r in rows if r["kw"]["anchor"] == anchor]
        best = max([r for r in cand if r["completed_all"] == 1.0] or cand, key=lambda r: r["score"])
        best["val"] = val_score([m for m, _ in ev.run("bead_geo", None, split_ids("val"), SEED, **best["kw"])])
        sel[anchor] = best
        print("SELECTED", anchor, best["kw"]["k"], "val", round(best["val"]["score"], 4), flush=True)
    (RUNS / "expert_k_tuning.json").write_text(json.dumps({"tuning_ids": ids, "selected": sel, "grid": rows}, indent=1))

    circ = split_ids("test", "f1tenth_circuit")
    synth = split_ids("test", "closed_shape")[:N_SYNTH] + split_ids("test", "open_curve")[:N_SYNTH]
    (OUT / "traj").mkdir(parents=True, exist_ok=True)
    out = []
    for anchor in ANCHORS:
        for k in KS:
            kw = {"ctrl": "expert_k", "k": k, "anchor": anchor}
            name = f"expert_k{k}_{anchor}"
            res = (ev.run("bead_geo", None, circ, EVAL_SEEDS, keep_traj=True, **kw)
                   + ev.run("bead_geo", None, synth, SYNTH_SEED, keep_traj=True, **kw))
            trajs = {}
            for m, extra in res:
                fam = "circuit" if m["id"].startswith("circuit_") else ("closed" if m["id"].startswith("closed") else "curve")
                row = {"controller": name, "track": "A+B", "plant": "bead", "train_seed": -1, "ref": m.pop("id"),
                       "family": fam, "eval_seed": m.pop("seed"), "k": k, "anchor": anchor, **m}
                out.append(row)
                trajs[f"{row['ref']}__{row['eval_seed']}"] = extra["traj"].astype(np.float32)
            np.savez_compressed(OUT / "traj" / f"{name}__sNone.npz", **trajs)
            print(f"{name:24s} {len(res)} episodes", flush=True)
    ev.close()
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(OUT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(out)
    print("wrote", OUT / "episodes.csv", len(out), "rows")


if __name__ == "__main__":
    main()
