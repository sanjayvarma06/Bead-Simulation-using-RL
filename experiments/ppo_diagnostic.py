"""Diagnostic for the PPO fine-tuning result (VALIDATION circuits only; no test data).

Question: did PPO fine-tuning fail to optimise its own objective, or did it optimise the
objective it is given -- the expected return of the *stochastic* policy (mean + Gaussian
exploration noise) -- while the *deterministic* mean used at evaluation became less accurate?

For each seed s in {0, 1}, both policies are run on the 4 validation circuits:
  * DAgger(s) mean + noise with std = exp(-0.5)  (exactly the policy PPO starts from)
  * DAgger->PPO(s) mean + noise with its own learned std   (the policy PPO ends with)
  * both deterministic (mean only)
with 3 noise seeds per circuit, recording the environment reward and the tracking RMSE.

Usage: python experiments/ppo_diagnostic.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, ROOT, split_ids, load_path, run_bead, actor_mean  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

NOISE_SEEDS = [11, 22, 33]
START_SEED = 777


def evaluate(model, std, refs, noise_seed):
    rng = np.random.default_rng(noise_seed)
    pol = (lambda o: actor_mean(model, o)) if std == 0 else (lambda o: actor_mean(model, o) + rng.normal(0, std, 2))
    out = []
    for r in refs:
        pts, brk = load_path(r)
        m, _ = run_bead(pol, pts, brk, START_SEED)
        out.append((m["env_reward"], m["rmse"], m["time_s"]))
    return np.array(out)


def main():
    torch.set_num_threads(1)
    refs = split_ids("val", "f1tenth_circuit")
    res = {}
    for s in (0, 1):
        dag = PPO.load(RUNS / "bead" / "dagger" / f"seed{s}" / "best.zip", device="cpu")
        dp = PPO.load(RUNS / "bead" / "dagger_ppo" / f"seed{s}" / "best.zip", device="cpu")
        std_start = float(np.exp(-0.5))
        std_end = float(torch.exp(dp.policy.log_std).mean())
        rows = {
            "dagger_det": evaluate(dag, 0, refs, 0),
            "dagger_ppo_det": evaluate(dp, 0, refs, 0),
            "dagger_stoch": np.mean([evaluate(dag, std_start, refs, n) for n in NOISE_SEEDS], axis=0),
            "dagger_ppo_stoch": np.mean([evaluate(dp, std_end, refs, n) for n in NOISE_SEEDS], axis=0),
        }
        res[f"seed{s}"] = {"std_start": std_start, "std_end": std_end,
                           **{k: {"reward": float(v[:, 0].mean()), "rmse_cm": float(100 * v[:, 1].mean()),
                                  "time_s": float(v[:, 2].mean())} for k, v in rows.items()}}
        print(json.dumps(res[f"seed{s}"], indent=1), flush=True)
    out = ROOT / "results" / "final" / "verification" / "ppo_diagnostic.json"
    out.write_text(json.dumps({"validation_circuits": refs, "noise_seeds": NOISE_SEEDS, **res}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
