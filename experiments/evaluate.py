"""Final evaluation on the TEST split. Every number is computed here from executed rollouts.

Outputs (results/final/):
  episodes.csv          one row per (controller, training seed, reference, evaluation seed)
  traj/<controller>__s<seed>.npz   raw trajectories (float32) for every row
  controllers.json      exact configuration / checkpoint of every evaluated controller

Usage: python experiments/evaluate.py [--only name1,name2]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, EVAL_SEEDS, split_ids  # noqa: E402
from valeval import Evaluator  # noqa: E402

OUT = ROOT / "results" / "final"
SEEDS = [0, 1]          # reduced from 5 by the user's compute budget
N_SYNTH = 50            # first 50 test references per synthetic family (of 150), same reason
SYNTH_SEED = [777]


def relative_paths(text: str) -> str:
    """Record checkpoint paths relative to the repository root (portable across machines)."""
    root = json.dumps(str(ROOT))[1:-1] + "\\\\"
    return text.replace(root, "").replace("\\\\", "/")


def controllers():
    tune = json.loads((RUNS / "pp_tuning.json").read_text())
    v_match = tune["v_match"]
    C = []
    add = lambda **k: C.append(k)  # noqa: E731
    add(name="expert", track="A+B", plant="bead", kind="bead", model="expert", seed=None, learned=False)
    for mode in ("dagger", "ppo", "dagger_ppo"):
        for s in SEEDS:
            add(name=f"bead_{mode}", track="A+B", plant="bead", kind="bead", seed=s, learned=True,
                model=str(RUNS / "bead" / mode / f"seed{s}" / "best.zip"))
    add(name="bead_published", track="A+B", plant="bead", kind="bead", seed=None, learned=True,
        model=str(ROOT / "results/models/PPO_BTP_MODEL.zip"),
        note="original checkpoint; trained and selected on Hockenheim/Montreal/YasMarina (test circuits)")
    add(name="pp_fixed_original", track="A", plant="bicycle", kind="pp", seed=None, learned=False, model=None,
        kw={"mode": "fixed", "fixed_L": 0.55, "fixed_g": 1.05})
    add(name="pp_adaptive_original", track="A", plant="bicycle", kind="pp", seed=None, learned=False, model=None,
        kw={"mode": "adaptive", "adapt_a": 0.35, "adapt_b": 0.25, "fixed_g": 1.05})
    for key, name, track in (("fixed_native", "pp_fixed_tuned", "A"), ("adaptive_native", "pp_adaptive_tuned", "A"),
                             ("fixed_matched", "pp_fixed_tuned_matched", "B"),
                             ("adaptive_matched", "pp_adaptive_tuned_matched", "B")):
        add(name=name, track=track, plant="bicycle", kind="pp", seed=None, learned=False, model=None,
            kw=tune[key]["kwargs"])
    for mode, sub, track, v in (("joint", "joint", "A", None), ("ld_only", "ld_only", "A", None),
                                ("joint", "joint_matched", "B", v_match)):
        for s in SEEDS:
            d = RUNS / "rlpp" / sub / f"seed{s}"
            add(name=f"rlpp_{sub}", track=track, plant="bicycle", kind="pp", seed=s, learned=True,
                model=str(d / "best.zip"), kw={"mode": mode, "v_cap": v, "vecnorm": str(d / "best_vecnorm.pkl")})
    add(name="rlpp_published", track="A", plant="bicycle", kind="pp", seed=None, learned=True,
        model=str(ROOT / "results/models/RL_PP_JOINT_MODEL.zip"), kw={"mode": "joint"},
        note="original checkpoint (trained on path1.jpeg); evaluated on raw observations as in the paper "
             "because its VecNormalize statistics were never saved")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traj").mkdir(exist_ok=True)
    circ = split_ids("test", "f1tenth_circuit")
    synth = split_ids("test", "closed_shape")[:N_SYNTH] + split_ids("test", "open_curve")[:N_SYNTH]
    C = controllers()
    if args.only:
        keep = set(args.only.split(","))
        C = [c for c in C if c["name"] in keep]
    for c in C:
        if c["model"] not in (None, "expert") and not Path(c["model"]).exists():
            raise FileNotFoundError(f"missing checkpoint for {c['name']} seed {c['seed']}: {c['model']}")
    ev = Evaluator(args.workers)
    csv_path = OUT / ("episodes_partial.csv" if args.only else "episodes.csv")
    rows_all = []
    t0 = time.time()
    for c in C:
        kw = dict(c.get("kw", {}), keep_traj=True)
        res = ev.run(c["kind"], c["model"], circ, EVAL_SEEDS, **kw) + ev.run(c["kind"], c["model"], synth, SYNTH_SEED, **kw)
        trajs = {}
        for m, extra in res:
            fam = "circuit" if m["id"].startswith("circuit_") else ("closed" if m["id"].startswith("closed") else "curve")
            row = {"controller": c["name"], "track": c["track"], "plant": c["plant"],
                   "train_seed": -1 if c["seed"] is None else c["seed"], "ref": m.pop("id"), "family": fam,
                   "eval_seed": m.pop("seed"), **m}
            rows_all.append(row)
            trajs[f"{row['ref']}__{row['eval_seed']}"] = extra["traj"].astype(np.float32)
        np.savez_compressed(OUT / "traj" / f"{c['name']}__s{c['seed']}.npz", **trajs)
        print(f"{c['name']:28s} seed {c['seed']}: {len(res)} episodes  ({time.time() - t0:.0f}s)", flush=True)
    ev.close()
    keys = []
    for r in rows_all:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows_all)
    if not args.only:
        (OUT / "controllers.json").write_text(relative_paths(json.dumps(C, indent=1, default=str)))
    print("wrote", csv_path, len(rows_all), "rows")


if __name__ == "__main__":
    main()
