"""PPO genuinely optimises actor and critic from the environment reward; checkpoints round-trip."""
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from bead_gym_env import BeadTraceEnv
from common import actor_mean


def straight():
    x = np.linspace(-4, 4, 80)
    return np.stack([x, np.zeros_like(x)], 1)


class ScaledReward(gym.RewardWrapper):
    def __init__(self, env, k):
        super().__init__(env); self.k = k
    def reward(self, r):
        return self.k * r


def make(k=1.0):
    return ScaledReward(BeadTraceEnv(paths=[(straight(), ())], seed=0), k)


def params(model):
    return {n: p.detach().clone() for n, p in model.policy.named_parameters()}


def small_ppo(env):
    return PPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=2, seed=0, device="cpu",
               policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))


def test_ppo_updates_actor_and_critic():
    m = small_ppo(make())
    before = params(m)
    m.learn(256)
    after = params(m)
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert any(n.startswith("mlp_extractor.policy_net") for n in changed)
    assert any(n.startswith("mlp_extractor.value_net") for n in changed)
    assert "action_net.weight" in changed and "value_net.weight" in changed and "log_std" in changed
    for key in ("train/clip_fraction", "train/approx_kl", "train/value_loss", "train/entropy_loss"):
        assert key in m.logger.name_to_value


def test_reward_drives_the_update():
    """Same seed, same data stream; only the reward differs -> different parameters."""
    a, b = small_ppo(make(1.0)), small_ppo(make(-1.0))
    a.learn(256); b.learn(256)
    pa, pb = params(a), params(b)
    assert not torch.equal(pa["action_net.weight"], pb["action_net.weight"])


def test_checkpoint_roundtrip_and_eval_has_no_side_effects(tmp_path):
    m = small_ppo(make())
    m.learn(128)
    m.save(tmp_path / "m.zip")
    r = PPO.load(tmp_path / "m.zip", device="cpu")
    obs = np.random.default_rng(0).uniform(-0.2, 0.2, (20, 16)).astype(np.float32)
    before = params(r)
    a1 = np.stack([actor_mean(m, o) for o in obs])
    a2 = np.stack([actor_mean(r, o) for o in obs])
    a3 = np.stack([actor_mean(r, o) for o in obs])
    assert np.allclose(a1, a2) and np.array_equal(a2, a3)
    assert all(torch.equal(before[n], p) for n, p in params(r).items())


def test_dagger_ppo_initialisation_copies_actor_exactly(tmp_path):
    """Replicates the weight transfer in train_bead.train_ppo: the new PPO model must output the
    same actor means as the DAgger checkpoint, while its critic is freshly initialised."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train_bead import new_model, set_actor_trainable
    env = DummyVecEnv([make])
    src = new_model(env, 1, -0.5)
    with torch.no_grad():                       # make the "DAgger" actor clearly non-default
        for p in src.policy.action_net.parameters():
            p.add_(0.3)
    src.save(tmp_path / "dagger.zip")
    loaded = PPO.load(tmp_path / "dagger.zip", device="cpu")
    dst = new_model(env, 2, -0.5)
    sd = {k: v for k, v in loaded.policy.state_dict().items()
          if k.startswith(("mlp_extractor.policy_net", "action_net"))}
    r = dst.policy.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys
    obs = np.random.default_rng(0).uniform(-0.3, 0.3, (32, 16)).astype(np.float32)
    assert np.allclose([actor_mean(dst, o) for o in obs], [actor_mean(src, o) for o in obs], atol=1e-6)
    assert not torch.equal(dst.policy.value_net.weight, src.policy.value_net.weight)
    # frozen actor: a PPO update must leave the actor unchanged but move the critic
    set_actor_trainable(dst, False)
    before = params(dst)
    dst.n_steps = 128; dst.batch_size = 64; dst.n_epochs = 1
    dst.rollout_buffer.buffer_size = 128; dst.rollout_buffer.reset()
    dst.learn(128)
    after = params(dst)
    for n in before:
        if n.startswith(("mlp_extractor.policy_net", "action_net", "log_std")):
            assert torch.equal(before[n], after[n]), n
    assert not torch.equal(before["value_net.weight"], after["value_net.weight"])
