"""Figures of the revised manuscript, built only from executed rollouts and training logs:
results/final/episodes.csv, results/final/traj/*.npz, runs/**/val_log.csv, datasets/path_cache.
Visual style is that of make_figures.py (the original manuscript).

Usage: python paper/scripts/make_figures_v2.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import load_path, load_manifest, run_pp, vecnorm_obs_fn  # noqa: E402

FINAL = ROOT / "results" / "final"
RUNS = ROOT / "runs"
FIG = ROOT / "paper" / "revised" / "figures"   # used directly by main.tex
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Colours follow make_figures.py; new controllers take further entries of the same cycle.
C = {"ref": "#555555", "expert": "#333333", "bead_dagger": "#2ca02c", "bead_dagger_ppo": "#17becf",
     "bead_ppo": "#8c564b", "bead_published": "#bcbd22", "rlpp_joint": "#1f77b4", "rlpp_ld_only": "#ff7f0e",
     "pp_adaptive_tuned": "#9467bd", "pp_fixed_tuned": "#d62728", "pp_fixed_original": "#e377c2",
     "rlpp_published": "#7f7f7f"}
LBL = {"expert": "Geometric expert", "bead_dagger": "Bead-DAgger", "bead_dagger_ppo": "Bead-DAgger→PPO",
       "bead_ppo": "Bead-PPO (scratch)", "bead_published": "Bead (published ckpt.)",
       "rlpp_joint": r"RL–PP (joint $L_d,g$)", "rlpp_ld_only": r"RL–PP ($L_d$ only)",
       "pp_adaptive_tuned": "Adaptive PP (tuned)", "pp_fixed_tuned": "Fixed PP (tuned)",
       "pp_fixed_original": "Fixed PP (orig. gains)", "rlpp_published": "RL–PP (published ckpt.)"}
MAIN = ["expert", "bead_dagger", "bead_dagger_ppo", "bead_ppo", "rlpp_joint", "rlpp_ld_only",
        "pp_fixed_tuned", "pp_adaptive_tuned"]

DF = pd.read_csv(FINAL / "episodes.csv")


def seed_means(family="circuit"):
    d = DF[DF.family == family]
    per = d.groupby(["controller", "train_seed", "ref"]).mean(numeric_only=True).reset_index()
    return per.groupby(["controller", "train_seed"]).mean(numeric_only=True).reset_index()


# ---------------------------------------------------------------- Fig. 1 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 90); ax.axis("off")
    bands = [("Data / Perception", "#0f7b7b", 61), ("Learning (DAgger, PPO, DAgger→PPO)", "#c9c9c9", 31),
             ("Evaluation / Benchmark", "#f5b700", 1)]
    for title, col, y in bands:
        ax.add_patch(Rectangle((0.5, y), 99, 28, color=col, zorder=0))
        ax.add_patch(Rectangle((22, y + 21.5), 56, 5, facecolor="white", edgecolor="#333", lw=0.5))
        ax.text(50, y + 24, title, ha="center", va="center", fontsize=6.4, weight="bold")

    def box(x, y, txt, fs=5.4):
        ax.add_patch(FancyBboxPatch((x, y), 28, 13, boxstyle="round,pad=0.6", facecolor="white",
                                    edgecolor="#333", lw=0.5))
        ax.text(x + 14, y + 6.5, txt, ha="center", va="center", fontsize=fs, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#1b1b1b", mutation_scale=6))

    rows = [("Raster image\nOtsu threshold\n+ morphology",
             "Zhang–Suen skeleton\n/ contour → ordered\npolyline $\\{p_i\\}$",
             "Arc-length resampling\n(350 pts) → 20×20 m\narena frame"),
            ("Bead environment\n16-D obs, 2-D accel.\naction, reward (3)",
             "Expert + DAgger\n(supervised) and/or\nPPO on reward (3)",
             "Selection on\nvalidation circuits\n+ shapes only"),
            ("Expert, bead policies,\nFixed/Adaptive PP,\nRL–PP ($L_d$; $L_d,g$)",
             "Common metrics:\nGeom@τ, RMSE, P95,\norder, time, speed",
             "8 held-out circuits\n+ 100 held-out shapes\n(native / matched speed)")]
    for (t0, t1, t2), (_, _, y) in zip(rows, bands):
        yb = y + 4
        box(3, yb, t0); box(36, yb, t1); box(69, yb, t2)
        arrow(31.8, yb + 6.5, 35.2, yb + 6.5); arrow(64.8, yb + 6.5, 68.2, yb + 6.5)
    arrow(83, 64.4, 83, 51.6); arrow(83, 34.4, 83, 21.6)
    fig.savefig(FIG / "fig1_pipeline.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 2 circuit split
def fig_split():
    ents = [e for e in load_manifest()["entries"] if e["family"] == "f1tenth_circuit"]
    order = {"train": 0, "val": 1, "test": 2}
    ents.sort(key=lambda e: (order[e["split"]], e["id"]))
    col = {"train": "#2ca02c", "val": "#ff7f0e", "test": "#1f77b4"}
    fig, axs = plt.subplots(3, 8, figsize=(3.5, 1.65))
    for ax in axs.flat:
        ax.axis("off")
    for ax, e in zip(axs.flat, ents):
        p, _ = load_path(e["id"])
        ax.plot(p[:, 0], p[:, 1], color=col[e["split"]], lw=0.7)
        ax.set_aspect("equal")
        short = {"MexicoCity": "Mexico", "MoscowRaceway": "Moscow", "Nuerburgring": "Nürburg.",
                 "Oschersleben": "Oschersl."}
        name = e["id"].replace("circuit_", "")
        ax.set_title(short.get(name, name), fontsize=4.6, pad=0.5)
    for s, c in col.items():
        axs.flat[-1].plot([], [], color=c, lw=1.2, label={"train": "train (11)", "val": "val (4)", "test": "test (8)"}[s])
    fig.legend(*axs.flat[-1].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.07), fontsize=6)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    fig.savefig(FIG / "fig2_split.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3 bars with seed range
def fig_bars():
    sm = seed_means("circuit")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for ax, m, sc, yl in ((axs[0], "geom010", 100, "Geom@0.10 [%] ↑"), (axs[1], "rmse", 100, "Tracking RMSE [cm] ↓")):
        for k, c in enumerate(MAIN):
            v = sm[sm.controller == c][m].to_numpy() * sc
            ax.bar(k, v.mean(), 0.75, color=C[c], label=LBL[c])
            if len(v) > 1:
                ax.plot([k, k], [v.min(), v.max()], color="k", lw=0.8)
                ax.scatter([k] * len(v), v, s=4, color="k", zorder=3)
        ax.set_xticks([]); ax.set_ylabel(yl); ax.grid(axis="y", alpha=0.3); ax.tick_params(length=2)
    axs[0].set_ylim(75, 100.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24),
               frameon=False, fontsize=5.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig3_bars.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4 precision vs pace
def fig_pace():
    sm = seed_means("circuit")
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    pts = MAIN + ["pp_fixed_tuned_matched", "pp_adaptive_tuned_matched", "rlpp_joint_matched"]
    for c in pts:
        s = sm[sm.controller == c]
        base = c.replace("_matched", "")
        matched = c.endswith("_matched")
        ax.scatter(s.time_s.mean(), 100 * s.rmse.mean(), s=22, color=C[base], marker="s" if matched else "o",
                   facecolors="none" if matched else C[base], lw=0.9,
                   label=LBL[base] + (" @ $v_{match}$" if matched else ""))
    ax.set_yscale("log"); ax.set_xlabel("Mean traversal time per test circuit [s]")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_ylabel("Tracking RMSE [cm] (log)"); ax.grid(alpha=0.3, which="both"); ax.tick_params(length=2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=5.6)
    fig.savefig(FIG / "fig4_pace.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 5 learning curves (validation)
def read_val(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_learning():
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.6), sharey=True)
    for mode, c in (("ppo", "bead_ppo"), ("dagger_ppo", "bead_dagger_ppo")):
        for s in (0, 1):
            r = read_val(RUNS / "bead" / mode / f"seed{s}" / "val_log.csv")
            x = [float(v["timesteps"]) / 1e3 for v in r]; y = [100 * float(v["rmse_circuits"]) for v in r]
            axs[0].plot(x, y, color=C[c], lw=0.9, marker="o", ms=2, label=LBL[c] if s == 0 else None)
    for s in (0, 1):
        r = read_val(RUNS / "bead" / "dagger" / f"seed{s}" / "val_log.csv")
        axs[1].plot([int(v["round"]) for v in r], [100 * float(v["rmse_circuits"]) for v in r], color=C["bead_dagger"],
                    lw=0.9, marker="o", ms=2, label=LBL["bead_dagger"] if s == 0 else None)
    ex = json.loads((RUNS / "expert_trainval.json").read_text())["val"]["circuit_rmse"] * 100
    for ax in axs:
        ax.axhline(ex, color=C["expert"], ls=":", lw=0.8, label="Expert")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.tick_params(length=2)
        from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10])); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.legend(frameon=False, fontsize=5.6, loc="center right")
    axs[0].legend(frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    axs[0].set_xlabel("PPO environment steps [k]"); axs[1].set_xlabel("DAgger round")
    axs[0].set_ylabel("Val. circuit RMSE [cm]")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(FIG / "fig5_learning.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 6 traces at a hairpin
def traj(name, seed, ref, ev=50000):
    return np.load(FINAL / "traj" / f"{name}__s{seed}.npz")[f"{ref}__{ev}"]


def fig_zoom():
    ref = "circuit_Montreal"
    p, _ = load_path(ref)
    # Zoom windows centred on the two (well separated) reference points that Bead-DAgger->PPO
    # misses by the largest distance, chosen from the data rather than by hand.
    from geometry_utils import points_to_polyline_distance
    dref = points_to_polyline_distance(p, traj("bead_dagger_ppo", 0, ref))
    # Exclude the start/finish region: the bead episode ends at 99 % progress, so the last
    # waypoints before loop closure are unvisited by construction (not a tracking error).
    dref[:8] = -1; dref[-8:] = -1
    i1 = int(np.argmax(dref))
    far = np.linalg.norm(p - p[i1], axis=1) > 3.0
    i2 = int(np.argmax(np.where(far, dref, -1)))
    h = 0.8
    wins = [(p[i, 0] - h, p[i, 0] + h, p[i, 1] - h, p[i, 1] + h) for i in (i1, i2)]
    sets = [("expert", None), ("bead_dagger", 0), ("bead_dagger_ppo", 0), ("bead_ppo", 0),
            ("pp_fixed_tuned", None), ("rlpp_joint", 0)]
    fig = plt.figure(figsize=(3.5, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.3)
    axf = fig.add_subplot(gs[0, 0])
    axf.plot(p[:, 0], p[:, 1], color=C["ref"], lw=0.7)
    for k, (x0, x1, y0, y1) in enumerate(wins):
        axf.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#d62728", ls="--", lw=0.8))
        axf.text(x1 + 0.3, y1, "(%s)" % "bc"[k], fontsize=6)
    axf.set_aspect("equal"); axf.set_xticks([]); axf.set_yticks([]); axf.set_title("(a) Montreal (test)")
    for k, (x0, x1, y0, y1) in enumerate(wins):
        ax = fig.add_subplot(gs[0, k + 1])
        ax.plot(p[:, 0], p[:, 1], color="#999", lw=4.5, alpha=0.5, label="Reference")
        for name, s in sets:
            t = traj(name, s, ref)
            ax.plot(t[:, 0], t[:, 1], color=C[name], lw=0.8, label=LBL[name])
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.tick_params(length=2, labelsize=6); ax.set_title(f"({'bc'[k]}) zoom")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5.6)
    fig.savefig(FIG / "fig6_zoom.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 7 learned RL-PP schedule
def fig_interp():
    d = RUNS / "rlpp" / "joint" / "seed0"
    from stable_baselines3 import PPO
    model = PPO.load(d / "best.zip", device="cpu")
    pts, brk = load_path("circuit_Montreal")
    m, ex = run_pp("joint", pts, brk, 50000, model=model, obs_norm=vecnorm_obs_fn(d / "best_vecnorm.pkl"))
    L, G = ex["L"], ex["G"]
    from pure_pursuit import compute_path_curvature
    k = np.abs(compute_path_curvature(pts, is_closed=True))
    tr = ex["traj"][1:]
    idx = np.argmin(((tr[:, None, :] - pts[None]) ** 2).sum(-1), axis=1)
    kmax = np.max(np.stack([k[(idx + o) % len(k)] for o in (0, 5, 12)]), axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.45))
    axs[0].scatter(kmax, L, s=1.5, color=C["rlpp_joint"], alpha=0.5, rasterized=True)
    axs[1].scatter(kmax, G, s=1.5, color=C["pp_adaptive_tuned"], alpha=0.5, rasterized=True)
    axs[0].set_ylabel("$L_d$ [m]"); axs[1].set_ylabel("$g$")
    for ax in axs:
        ax.set_xlabel(r"$\kappa^{\max}$ [m$^{-1}$]"); ax.grid(alpha=0.3); ax.tick_params(length=2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig7_interp.png"); plt.close(fig)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    stats = {"L_min": float(L.min()), "L_max": float(L.max()), "L_mean": float(L.mean()),
             "g_min": float(G.min()), "g_max": float(G.max()), "g_std": float(G.std()),
             "r_L_kmax": r(L, kmax), "r_g_kmax": r(G, kmax)}
    (FIG / "fig7_stats.json").write_text(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- Fig. 8 image -> reference
def fig_inputs():
    """Raster inputs (top) and the ordered references extracted from them (bottom)."""
    import cv2
    from environment import BeadEnvironment
    items = [("(a) Montreal circuit", ROOT / "datasets/f1tenth_circuits/images/Montreal.png"),
             ("(b) closed shape", ROOT / "datasets/closed_shapes_dataset/test/shape_test_0001.jpg"),
             ("(c) open curve", ROOT / "datasets/curve_dataset/test/curve_test_0002.jpg"),
             ("(d) hand drawing", ROOT / "datasets/test_images/path1.jpeg")]
    fig, axs = plt.subplots(2, 4, figsize=(3.5, 1.95), gridspec_kw={"hspace": 0.12, "wspace": 0.06})
    for k, (title, p) in enumerate(items):
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        axs[0, k].imshow(im); axs[0, k].set_title(title, fontsize=6, pad=2)
        env = BeadEnvironment(image_path=str(p), strict=True, seed=0)
        pts = env.path_points
        sc = axs[1, k].scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="viridis", s=0.6, lw=0)
        axs[1, k].scatter(*pts[0], s=9, color="k", zorder=3)
        axs[1, k].set_aspect("equal")
        for ax in axs[:, k]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
    cb = fig.colorbar(sc, ax=axs[1, :], orientation="horizontal", fraction=0.06, pad=0.04, aspect=40)
    cb.set_label("waypoint index (start = black dot)", fontsize=6); cb.ax.tick_params(labelsize=5.5, length=1.5)
    fig.savefig(FIG / "fig8_inputs.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 9 full-lap traces
def fig_traces():
    from geometry_utils import points_to_polyline_distance
    refs = [("Hockenheim", "circuit_Hockenheim"), ("Montreal", "circuit_Montreal"), ("Yas Marina", "circuit_YasMarina")]
    ctrls = [("bead_dagger", 0), ("bead_dagger_ppo", 0), ("pp_fixed_tuned", None)]
    fig, axs = plt.subplots(3, 3, figsize=(3.5, 2.75), gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    for i, (name, ref) in enumerate(refs):
        p, _ = load_path(ref)
        for j, (c, s) in enumerate(ctrls):
            ax = axs[i, j]
            t = traj(c, s, ref)
            dref = points_to_polyline_distance(p, t)
            dref[-8:] = 0          # start/finish closure, see text
            ax.plot(p[:, 0], p[:, 1], color="#bbbbbb", lw=2.4, solid_capstyle="round")
            ax.plot(t[:, 0], t[:, 1], color=C[c], lw=0.6)
            bad = dref > 0.10
            ax.scatter(p[bad, 0], p[bad, 1], marker="x", s=6, lw=0.5, color="#d62728", zorder=3)
            ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(LBL[c], fontsize=6.2, pad=2)
            if j == 0:
                ax.set_ylabel(name, fontsize=6.5)
            ax.text(0.97, 0.04, f"{100 * np.sqrt(np.mean(points_to_polyline_distance(t, p) ** 2)):.2f} cm",
                    transform=ax.transAxes, fontsize=5.5, ha="right",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    fig.savefig(FIG / "fig9_traces.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 10 synthetic contours
def fig_shapes():
    ids = ["closed_shape_shape_test_0001", "closed_shape_shape_test_0003",
           "open_curve_curve_test_0002", "open_curve_curve_test_0005"]
    ctrls = [("bead_dagger", 0), ("bead_dagger_ppo", 0), ("bead_ppo", 0), ("pp_fixed_tuned", None)]
    fig, axs = plt.subplots(1, 4, figsize=(3.5, 1.05), gridspec_kw={"wspace": 0.06})
    for ax, ref in zip(axs, ids):
        p, _ = load_path(ref)
        ax.plot(p[:, 0], p[:, 1], color="#bbbbbb", lw=2.6)
        for c, s in ctrls:
            t = traj(c, s, ref, ev=777)
            ax.plot(t[:, 0], t[:, 1], color=C[c], lw=0.6, label=LBL[c])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5.4,
               handlelength=1.2, columnspacing=0.8)
    fig.savefig(FIG / "fig10_shapes.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 11 error CDF
def fig_cdf():
    from geometry_utils import points_to_polyline_distance
    ctrls = [("expert", None), ("bead_dagger", 0), ("bead_dagger_ppo", 0), ("bead_ppo", 0),
             ("rlpp_joint", 0), ("pp_fixed_tuned", None), ("pp_adaptive_tuned", None)]
    from common import split_ids
    refs = split_ids("test", "f1tenth_circuit")
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for c, s in ctrls:
        e = np.concatenate([points_to_polyline_distance(traj(c, s, r), load_path(r)[0]) for r in refs]) * 100
        e = np.sort(e)
        ax.plot(e, np.linspace(0, 1, len(e)), color=C[c], lw=1.0, label=LBL[c])
    ax.axvline(10, color="k", ls=":", lw=0.6)
    ax.set_xlim(0, 15); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.tick_params(length=2)
    ax.set_xlabel("Lateral error $e$ [cm] (all 8 test circuits, first episode)"); ax.set_ylabel(r"$\Pr(E\leq e)$")
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, ncol=2)
    fig.savefig(FIG / "fig11_cdf.png"); plt.close(fig)


if __name__ == "__main__":
    only = set(sys.argv[1:])          # e.g. "python make_figures_v2.py fig_cdf fig_traces"
    for f in (fig_pipeline, fig_split, fig_bars, fig_pace, fig_learning, fig_zoom, fig_interp,
              fig_inputs, fig_traces, fig_cdf):
        if not only or f.__name__ in only:
            f()
    print("figures ->", FIG)
