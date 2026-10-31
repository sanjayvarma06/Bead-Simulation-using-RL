"""Diagnostics for the PPO results (VALIDATION circuits only; no test data).

1. Reward versus accuracy. The geometric expert is run with Gaussian action noise of std
   0, exp(-2) and exp(-0.5) (3 noise seeds each); its return under the base reward and under the
   per-progress reward (reward_mode="progress") is recorded with its RMSE and lap time.
2. Fine-tuning. For every seed s and fine-tuning variant v in {dagger_ppo, dagger_ppo_acc}:
     DAgger(s) mean, and mean + noise with the initial std of the fine-tuning (exp(-2));
     DAgger->PPO(s) mean, and mean + noise with its own learned std;
   with the return under the reward the variant was trained on, RMSE and lap time.

Usage: python experiments/ppo_diagnostic.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, ROOT, split_ids, load_path, run_bead, actor_mean, expert_action  # noqa: E402
from evaluate import SEEDS  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

NOISE_SEEDS = [11, 22, 33]
START_SEED = 777
VARIANTS = {"dagger_ppo": "base", "dagger_ppo_acc": "progress"}


class NoisyExpert:
    def __init__(self, std, seed):
        self.std, self.rng = std, np.random.default_rng(seed)

    def core_action(self, core):
        a = expert_action(core)
        return a + self.rng.normal(0, self.std, 2) if self.std > 0 else a


def run_all(policy_factory, refs, reward_mode):
    out = []
    for r in refs:
        pts, brk = load_path(r)
        m, _ = run_bead(policy_factory(), pts, brk, START_SEED, reward_mode=reward_mode)
        out.append((m["env_reward"], 100 * m["rmse"], m["time_s"]))
    return np.array(out).mean(axis=0)


def summary(v):
    return {"reward": float(v[0]), "rmse_cm": float(v[1]), "time_s": float(v[2])}


def model_policy(model, std, seed):
    rng = np.random.default_rng(seed)
    if std == 0:
        return lambda: (lambda o: actor_mean(model, o))
    return lambda: (lambda o: actor_mean(model, o) + rng.normal(0, std, 2))


def main():
    torch.set_num_threads(1)
    refs = split_ids("val", "f1tenth_circuit")
    res = {"validation_circuits": refs, "noise_seeds": NOISE_SEEDS, "expert": {}, "finetune": {}}
    for std in (0.0, float(np.exp(-2.0)), float(np.exp(-0.5))):
        seeds = [0] if std == 0 else NOISE_SEEDS
        row = {}
        for mode in ("base", "progress"):
            row[mode] = summary(np.mean([run_all(lambda: NoisyExpert(std, n), refs, mode) for n in seeds], axis=0))
        res["expert"][f"{std:.3f}"] = row
        print("expert", round(std, 3), json.dumps(row), flush=True)
    std0 = float(np.exp(-2.0))
    for v, mode in VARIANTS.items():
        for s in SEEDS:
            dag = PPO.load(RUNS / "bead" / "dagger" / f"seed{s}" / "best.zip", device="cpu")
            dp = PPO.load(RUNS / "bead" / v / f"seed{s}" / "best.zip", device="cpu")
            std_end = float(torch.exp(dp.policy.log_std).mean())
            row = {"std_start": std0, "std_end": std_end, "reward_mode": mode,
                   "dagger_det": summary(run_all(model_policy(dag, 0, 0), refs, mode)),
                   "finetuned_det": summary(run_all(model_policy(dp, 0, 0), refs, mode)),
                   "dagger_stoch": summary(np.mean([run_all(model_policy(dag, std0, n), refs, mode)
                                                    for n in NOISE_SEEDS], axis=0)),
                   "finetuned_stoch": summary(np.mean([run_all(model_policy(dp, std_end, n), refs, mode)
                                                       for n in NOISE_SEEDS], axis=0))}
            res["finetune"][f"{v}/seed{s}"] = row
            print(v, s, json.dumps(row), flush=True)
    out = ROOT / "results" / "final" / "verification" / "ppo_diagnostic.json"
    out.write_text(json.dumps(res, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
