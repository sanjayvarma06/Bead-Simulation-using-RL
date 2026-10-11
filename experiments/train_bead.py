"""Training of the direct-action bead policy.

Three explicitly separated procedures (same network, same environment, same reward):

  dagger      Expert anchoring + DAgger (Algorithm 1 of the original paper), i.e. supervised
              regression of the actor mean onto expert labels. The reward is NOT used.
  ppo         Proximal Policy Optimization from a random initialisation, driven only by the
              environment reward (Eq. 3): rollouts, GAE, clipped surrogate, value loss, entropy.
  dagger_ppo  Actor initialised from the best `dagger` checkpoint of the same seed, critic
              warm-up with the actor frozen, then PPO fine-tuning on the reward.

Model selection uses validation references only (`val_score`), never the test split.

Examples:
  python experiments/train_bead.py --mode dagger --seed 0
  python experiments/train_bead.py --mode ppo --seed 0 --timesteps 3000000
  python experiments/train_bead.py --mode dagger_ppo --seed 0 --timesteps 3000000
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, split_ids, load_paths, expert_action, actor_mean  # noqa: E402
from bead_gym_env import BeadTraceEnv  # noqa: E402
from valeval import Evaluator, val_score  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402

VAL_SEEDS = [777]

# PPO settings: the "network configuration" column of Table II of the original paper,
# which was declared but never exercised. log_std_init is the only addition (see README).
PPO_KW = dict(learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5)
NET = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh)


def training_pool():
    """Training references; circuits are duplicated so they make up ~half of all episodes."""
    circ = load_paths(split_ids("train", "f1tenth_circuit"))
    syn = load_paths(split_ids("train", "closed_shape") + split_ids("train", "open_curve"))
    reps = max(1, round(len(syn) / max(1, len(circ))))
    return circ * reps + syn, circ, syn


def val_ids():
    return split_ids("val")


def make_env_fn(pool, seed):
    def _f():
        return BeadTraceEnv(paths=pool, seed=seed)
    return _f


def new_model(env, seed, log_std_init):
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
               policy_kwargs=dict(**NET, log_std_init=log_std_init), **PPO_KW)


# ----------------------------------------------------------------------------- DAgger
def _dagger_episode(args):
    model_path, pts, brk, beta, seed, use_expert_only = args
    torch.set_num_threads(1)
    from stable_baselines3 import PPO as _PPO
    model = None if use_expert_only else _PPO.load(model_path, device="cpu")
    env = BeadTraceEnv(paths=[(pts, brk)], seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    O, A = [], []
    done = trunc = False
    while not (done or trunc):
        ea = expert_action(env._env)
        O.append(obs.copy()); A.append(ea)
        act = ea if (use_expert_only or rng.random() < beta) else actor_mean(model, obs)
        obs, _, done, trunc, _ = env.step(act)
    return np.asarray(O, np.float32), np.asarray(A, np.float32)


def fit_actor(model, obs, acts, epochs, lr, batch_size=256):
    """MSE regression of the actor mean (features -> policy MLP -> action head)."""
    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-6)
    x = torch.as_tensor(obs); y = torch.as_tensor(acts)
    n = len(x); last = 0.0
    pol.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for st in range(0, n, batch_size):
            idx = perm[st:st + batch_size]
            pred = pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(x[idx])))
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
    pol.eval()
    return last


def train_dagger(args, out: Path, ev: Evaluator):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    _, circ, syn = training_pool()
    refs = circ + syn
    env = DummyVecEnv([make_env_fn(refs, args.seed)])
    model = new_model(env, args.seed, args.log_std_init)
    from concurrent.futures import ProcessPoolExecutor
    tmp = out / "_current.zip"
    log = []
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [(None, p, b, 1.0, args.seed * 100000 + 10 * i + k, True)
                for i, (p, b) in enumerate(refs) for k in range(args.expert_eps)]
        data = list(pool.map(_dagger_episode, jobs))
        O = np.concatenate([d[0] for d in data]); A = np.concatenate([d[1] for d in data])
        mse = fit_actor(model, O, A, epochs=12, lr=2e-4)
        betas = [0.40, 0.20, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00]
        best = -np.inf
        for rd in range(1, args.rounds + 1):
            model.save(tmp)
            jobs = [(str(tmp), p, b, betas[rd - 1], args.seed * 100000 + 1000 * rd + i, False)
                    for i, (p, b) in enumerate(refs)]
            data = list(pool.map(_dagger_episode, jobs))
            O = np.concatenate([O] + [d[0] for d in data]); A = np.concatenate([A] + [d[1] for d in data])
            mse = fit_actor(model, O, A, epochs=8, lr=1.5e-4 if rd <= 3 else 7e-5)
            model.save(out / "last.zip")
            vs = val_score([r for r, _ in ev.run("bead", out / "last.zip", val_ids(), VAL_SEEDS)])
            row = {"round": rd, "beta": betas[rd - 1], "dataset": len(O), "mse": mse, **vs}
            if vs["score"] > best:
                best = vs["score"]; model.save(out / "best.zip"); row["saved_best"] = True
            log.append(row)
            print(json.dumps(row), flush=True)
    tmp.unlink(missing_ok=True)
    write_csv(out / "val_log.csv", log)
    env.close()


# ----------------------------------------------------------------------------- PPO
class TrainLogger(BaseCallback):
    """Per-episode reward components and validation-based checkpointing."""

    def __init__(self, out: Path, ev: Evaluator, eval_every: int, freeze_until: int = 0):
        super().__init__()
        self.out, self.ev, self.eval_every, self.freeze_until = out, ev, eval_every, freeze_until
        self.next_eval = eval_every
        self.best = -np.inf
        self.acc: dict = {}
        self.episodes: list = []
        self.val_log: list = []
        self.frozen = freeze_until > 0

    def _on_training_start(self):
        if self.frozen:
            set_actor_trainable(self.model, False)

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            a = self.acc.setdefault(i, {})
            for k, v in info.get("reward_components", {}).items():
                a[k] = a.get(k, 0.0) + float(v)
            a["_len"] = a.get("_len", 0) + 1
            if done:
                a["coverage"] = info.get("coverage", 0.0)
                a["timestep"] = self.num_timesteps
                self.episodes.append(a)
                self.acc[i] = {}
        if self.frozen and self.num_timesteps >= self.freeze_until:
            set_actor_trainable(self.model, True)
            self.frozen = False
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self.validate()
        return True

    def _on_rollout_end(self):
        if self.episodes:
            for k in self.episodes[0]:
                self.logger.record(f"episode/{k}", float(np.mean([e.get(k, 0.0) for e in self.episodes])))
            self.logger.record("episode/count", len(self.episodes))
            self.episodes = []
        self.logger.record("train/actor_frozen", int(self.frozen))

    def validate(self):
        self.model.save(self.out / "last.zip")
        vs = val_score([r for r, _ in self.ev.run("bead", self.out / "last.zip", val_ids(), VAL_SEEDS)])
        row = {"timesteps": self.num_timesteps, "actor_frozen": self.frozen, **vs}
        # During critic warm-up the actor is unchanged, so it is not a new candidate.
        if not self.frozen and vs["score"] > self.best:
            self.best = vs["score"]; self.model.save(self.out / "best.zip"); row["saved_best"] = True
        self.val_log.append(row)
        write_csv(self.out / "val_log.csv", self.val_log)
        print(json.dumps(row), flush=True)


def set_actor_trainable(model, flag: bool):
    pol = model.policy
    for p in list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()) + [pol.log_std]:
        p.requires_grad_(flag)


def train_ppo(args, out: Path, ev: Evaluator):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    pool, _, _ = training_pool()
    venv = SubprocVecEnv([make_env_fn(pool, args.seed * 1000 + i) for i in range(args.n_envs)])
    venv = VecMonitor(venv)
    venv.seed(args.seed * 1000)
    model = new_model(venv, args.seed, args.log_std_init)
    freeze = 0
    if args.mode == "dagger_ppo":
        init = Path(args.init or RUNS / "bead" / "dagger" / f"seed{args.seed}" / "best.zip")
        src = PPO.load(init, device="cpu")
        sd = {k: v for k, v in src.policy.state_dict().items()
              if k.startswith(("mlp_extractor.policy_net", "action_net"))}
        missing = model.policy.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys
        (out / "init.txt").write_text(f"actor initialised from {init}\n")
        freeze = args.critic_warmup
    model.set_logger(configure(str(out), ["csv"]))
    cb = TrainLogger(out, ev, args.eval_every, freeze_until=freeze)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    cb.validate()
    model.save(out / "final.zip")
    (out / "timing.json").write_text(json.dumps({"wall_s": time.time() - t0, "timesteps": model.num_timesteps}))
    venv.close()


def write_csv(path: Path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dagger", "ppo", "dagger_ppo"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=250_000)
    ap.add_argument("--log_std_init", type=float, default=-1.0)
    ap.add_argument("--critic_warmup", type=int, default=300_000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--expert_eps", type=int, default=1)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)   # fastest for these small MLPs; lets runs share the CPU
    out = RUNS / "bead" / (args.mode + args.tag) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "ppo": PPO_KW, "net": {"pi": [256, 256], "vf": [256, 256], "act": "tanh"},
           "python": platform.python_version(), "torch": torch.__version__,
           "train_ids": split_ids("train"), "val_ids": val_ids()}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    ev = Evaluator(args.workers)
    try:
        if args.mode == "dagger":
            train_dagger(args, out, ev)
        else:
            train_ppo(args, out, ev)
    finally:
        ev.close()


if __name__ == "__main__":
    main()
