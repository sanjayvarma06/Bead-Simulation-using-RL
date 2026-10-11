"""Retraining of the RL-tuned Pure Pursuit baseline [4] on the training split.

Hyper-parameters are exactly those of `research_paper_code/train_rl_pure_pursuit.py`
(Table II, right column). Changes relative to that script:
  * training references = the training split (was one hand-drawn contour);
  * VecNormalize statistics are saved with every checkpoint and applied at evaluation
    (the original script trained on normalised observations but selected checkpoints and
    was evaluated on raw observations, because the statistics were never saved);
  * checkpoint selection on validation references with the shared `val_score`;
  * `--mode ld_only` trains a separate lookahead-only policy (training-time ablation).

Usage: python experiments/train_rlpp.py --mode joint --seed 0 [--matched]
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids, load_paths  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402
from train_bead import training_pool, write_csv  # noqa: E402
from rl_pure_pursuit_env import RLPurePursuitEnv  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor  # noqa: E402

VAL_SEEDS = [777]


def save_stats(venv: VecNormalize, path: Path):
    with open(path, "wb") as f:
        pickle.dump({"mean": venv.obs_rms.mean.copy(), "var": venv.obs_rms.var.copy(),
                     "clip_obs": venv.clip_obs, "epsilon": venv.epsilon}, f)


class ValCallback(BaseCallback):
    def __init__(self, out, ev, every, mode, v_cap):
        super().__init__()
        self.out, self.ev, self.every, self.mode, self.v_cap = out, ev, every, mode, v_cap
        self.next, self.best, self.log = every, -np.inf, []

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.validate()
        return True

    def validate(self):
        self.model.save(self.out / "last.zip")
        save_stats(self.model.get_env(), self.out / "last_vecnorm.pkl")
        rows = [r for r, _ in self.ev.run("pp", self.out / "last.zip", split_ids("val"), VAL_SEEDS,
                                          mode=self.mode, v_cap=self.v_cap,
                                          vecnorm=str(self.out / "last_vecnorm.pkl"))]
        vs = val_score(rows)
        row = {"timesteps": self.num_timesteps, **vs}
        if vs["score"] > self.best:
            self.best = vs["score"]
            self.model.save(self.out / "best.zip")
            save_stats(self.model.get_env(), self.out / "best_vecnorm.pkl")
            row["saved_best"] = True
        self.log.append(row)
        write_csv(self.out / "val_log.csv", self.log)
        print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["joint", "ld_only"], default="joint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--eval_every", type=int, default=50_000)
    ap.add_argument("--matched", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    v_cap = json.loads((RUNS / "expert_trainval.json").read_text())["v_match"] if args.matched else None
    out = RUNS / "rlpp" / (args.mode + ("_matched" if args.matched else "")) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    pool, _, _ = training_pool()
    (out / "config.json").write_text(json.dumps({**vars(args), "v_cap": v_cap,
                                                 "train_ids": split_ids("train")}, indent=1))

    venv = DummyVecEnv([lambda: RLPurePursuitEnv(mode=args.mode, seed=args.seed, paths=pool, v_cap=v_cap)])
    venv = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=30.0)
    model = PPO("MlpPolicy", venv, learning_rate=lambda f: 2.4e-4 * f, n_steps=2048, batch_size=256,
                n_epochs=5, gamma=0.99, gae_lambda=0.98, clip_range=0.2, target_kl=0.015, ent_coef=0.02,
                vf_coef=0.6, max_grad_norm=0.7, seed=args.seed, device="cpu", verbose=0)
    model.set_logger(configure(str(out), ["csv"]))
    ev = Evaluator(args.workers)
    try:
        cb = ValCallback(out, ev, args.eval_every, args.mode, v_cap)
        model.learn(total_timesteps=args.timesteps, callback=cb)
        cb.validate()
    finally:
        ev.close()


if __name__ == "__main__":
    main()
