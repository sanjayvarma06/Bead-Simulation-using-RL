"""Dynamic single-track vehicle model with linear tyres, ported from F1TENTH Gym.

Source: f1tenth/f1tenth_gym, gym/f110_gym/envs/dynamic_models.py, function vehicle_dynamics_st
(commit 4fdb9c7, 2025-06-09), which implements the CommonRoad single-track model. The right-hand
side below is the same expression; default parameters are the F1TENTH Gym defaults
(f110_env.py). Its wheelbase lf + lr = 0.3302 m matches the 0.33 m kinematic car of rl_pure_pursuit_env.

Deviations from F1TENTH Gym, so that the model can drive the arena-scaled circuits:
  * steering-angle limit +-1.10 rad (as the kinematic car) instead of +-0.4189 rad: the arena
    circuits have curvatures up to 5 m^-1, which need about 1.03 rad of steering;
  * integration with classical RK4, two sub-steps per 0.016 s control step (Gym: Euler or RK4 at 0.01 s).
The steering-rate limit (3.2 rad/s, bang-bang towards the commanded angle) and the switch to the
kinematic model below 0.5 m/s are kept as in F1TENTH Gym.

State x = [x, y, delta, v, psi, psi_dot, beta] (position of the centre of gravity, steering angle,
speed, yaw, yaw rate, side-slip angle at the centre of gravity).
"""
from __future__ import annotations
import numpy as np

F1TENTH_PARAMS = {"mu": 1.0489, "C_Sf": 4.718, "C_Sr": 5.4562, "lf": 0.15875, "lr": 0.17145, "h": 0.074,
                  "m": 3.74, "I": 0.04712, "sv_max": 3.2}
G = 9.81


def rhs(x, sv, acc, p=F1TENTH_PARAMS):
    mu, C_Sf, C_Sr, lf, lr, h, m, I = (p[k] for k in ("mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I"))
    d, v, psi, r, beta = x[2], x[3], x[4], x[5], x[6]
    if abs(v) < 0.5:                       # kinematic model at low speed (as in F1TENTH Gym)
        lwb = lf + lr
        return np.array([v * np.cos(psi), v * np.sin(psi), sv, acc, v / lwb * np.tan(d),
                         acc / lwb * np.tan(d) + v / (lwb * np.cos(d) ** 2) * sv, 0.0])
    Ff = C_Sf * (G * lr - acc * h)
    Fr = C_Sr * (G * lf + acc * h)
    L = lr + lf
    return np.array([
        v * np.cos(beta + psi),
        v * np.sin(beta + psi),
        sv,
        acc,
        r,
        -mu * m / (v * I * L) * (lf ** 2 * Ff + lr ** 2 * Fr) * r
        + mu * m / (I * L) * (lr * Fr - lf * Ff) * beta
        + mu * m / (I * L) * lf * Ff * d,
        (mu / (v ** 2 * L) * (Fr * lr - Ff * lf) - 1) * r
        - mu / (v * L) * (Fr + Ff) * beta
        + mu / (v * L) * Ff * d,
    ])


def step(x, delta_cmd, acc, dt, max_steer=1.10, p=F1TENTH_PARAMS, substeps=2):
    """Advance the state by dt with RK4; the steering angle moves towards delta_cmd at <= sv_max."""
    h = dt / substeps
    x = np.asarray(x, float).copy()
    for _ in range(substeps):
        diff = float(np.clip(delta_cmd, -max_steer, max_steer)) - x[2]
        sv = 0.0 if abs(diff) < 1e-4 else float(np.sign(diff)) * p["sv_max"]
        sv = float(np.clip(sv, -abs(diff) / h, abs(diff) / h))    # do not overshoot the command
        k1 = rhs(x, sv, acc, p)
        k2 = rhs(x + 0.5 * h * k1, sv, acc, p)
        k3 = rhs(x + 0.5 * h * k2, sv, acc, p)
        k4 = rhs(x + h * k3, sv, acc, p)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[2] = float(np.clip(x[2], -max_steer, max_steer))
    return x
