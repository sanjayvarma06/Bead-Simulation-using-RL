"""PPO with behaviour-cloning term, expert labels from observations, reward variants, shared lap rule."""
import numpy as np
import torch
from stable_baselines3 import PPO
from bead_gym_env import BeadTraceEnv
from environment import BeadEnvironment
from common import expert_action, run_bead, run_pp
from train_bead import PPOBC, expert_from_obs


def circle(n=300, r=4.0):
    t = np.linspace(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def small(cls, **kw):
    env = BeadTraceEnv(paths=[(circle(), ())], seed=0)
    return cls("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=2, seed=0, device="cpu",
               policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])), **kw)


def params(m):
    return {n: p.detach().clone() for n, p in m.policy.named_parameters()}


def test_ppobc_with_zero_weight_equals_ppo():
    # built and trained one after the other: learn() samples from the global torch stream,
    # which each constructor reseeds
    a = small(PPO); a.learn(256)
    b = small(PPOBC, bc_coef=0.0); b.learn(256)
    pa, pb = params(a), params(b)
    assert all(torch.equal(pa[n], pb[n]) for n in pa)


def test_bc_term_pulls_actor_towards_expert():
    a = small(PPOBC, bc_coef=0.0); a.learn(1024)
    b = small(PPOBC, bc_coef=50.0); b.learn(1024)
    env = BeadTraceEnv(paths=[(circle(), ())], seed=3)
    obs, _ = env.reset(seed=3)
    o = torch.as_tensor(np.stack([env.step(np.zeros(2))[0] for _ in range(50)]))
    def bc(m):
        pol = m.policy
        with torch.no_grad():
            mu = pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(o)))
        return float(((mu - expert_from_obs(o)) ** 2).mean())
    assert bc(b) < bc(a)
    assert "train/bc_loss" in b.logger.name_to_value


def test_expert_label_recovered_from_observation():
    env = BeadTraceEnv(paths=[(circle(), ())], seed=1)
    obs, _ = env.reset(seed=1)
    for _ in range(200):
        a = expert_action(env._env)
        lab = expert_from_obs(torch.as_tensor(obs[None]))[0].numpy()
        assert np.allclose(lab, a, atol=1e-5)
        obs, *_ = env.step(a)


def test_base_reward_unchanged_and_progress_mode_differs():
    pts = circle()
    a, _ = run_bead("expert", pts, (), seed=2)
    e1 = BeadTraceEnv(paths=[(pts, ())], seed=2, reward_mode="base")
    e2 = BeadTraceEnv(paths=[(pts, ())], seed=2, reward_mode="progress")
    e1.reset(seed=2); e2.reset(seed=2)
    r1 = r2 = 0.0
    done = False
    while not done:
        u = expert_action(e1._env)
        _, ra, d1, t1, _ = e1.step(u)
        _, rb, _, _, _ = e2.step(u)
        r1 += ra; r2 += rb; done = d1 or t1
    assert np.isclose(r1, a["env_reward"])
    assert not np.isclose(r1, r2)


def test_progress_precision_is_zero_without_progress():
    e = BeadEnvironment(paths=[(circle(), ())], seed=0, reward_mode="progress")
    e.reset(seed=0)
    e.pos_x, e.pos_y = 9.0, 9.0                      # far from the path: counter cannot advance
    _, _, _, _, info = e.step(np.zeros(2))
    assert info["reward_components"]["precision"] == 0.0


def test_car_and_bead_share_lap_rule():
    pts = circle()
    mb, _ = run_bead("expert", pts, (), seed=0)
    mc, _ = run_pp("fixed", pts, (), 0, fixed_L=0.3, fixed_g=1.0)
    assert mb["terminated"] and mc["terminated"]
    assert mb["geom010"] == mc["geom010"]            # both stop at the same waypoint
