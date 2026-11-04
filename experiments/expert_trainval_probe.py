"""Runs the geometric expert on train/val references (never test) to (a) confirm that the
step cap allows completion and (b) derive the matched speed for the controlled comparison."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import split_ids, RUNS
from valeval import Evaluator

if __name__ == "__main__":
    ev = Evaluator(16)
    out = {}
    for split in ("train", "val"):
        rows = [r for r, _ in ev.run("bead", "expert", split_ids(split), [777])]
        circ = [r for r in rows if r["id"].startswith("circuit_")]
        out[split] = {"n": len(rows), "completed": float(np.mean([r["completed"] for r in rows])),
                      "circuit_mean_speed": float(np.mean([r["mean_speed"] for r in circ])),
                      "circuit_geom010": float(np.mean([r["geom010"] for r in circ])),
                      "circuit_rmse": float(np.mean([r["rmse"] for r in circ])),
                      "synth_geom010": float(np.mean([r["geom010"] for r in rows if r not in circ])),
                      "synth_rmse": float(np.mean([r["rmse"] for r in rows if r not in circ])),
                      "not_completed": [r["id"] for r in rows if not r["completed"]]}
    ev.close()
    out["v_match"] = round(out["train"]["circuit_mean_speed"], 3)
    (RUNS).mkdir(exist_ok=True)
    (RUNS / "expert_trainval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
