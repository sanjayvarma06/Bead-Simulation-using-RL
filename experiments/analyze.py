"""Aggregates results/final/episodes.csv into tables and the pre-declared statistics.

Aggregation (learned controllers): episode -> mean over evaluation episodes per (reference,
training seed) -> mean over references per training seed -> mean +- std over training seeds.
Non-learned controllers are deterministic given the start seed and have one "seed".

Statistics (docs/02_PHASE2_PLAN.md, section F): paired over the 8 test circuits, exact two-sided
Wilcoxon signed-rank, mean difference with a 95 % bootstrap CI (circuits resampled; training seeds
resampled within learned controllers), Holm correction across the 14 pre-declared tests.

Usage: python experiments/analyze.py
"""
from __future__ import annotations
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

FINAL = ROOT / "results" / "final"
METRICS = ["geom005", "geom010", "rmse", "p95", "max_err", "off_path_frac", "sequential", "ordered_cov",
           "completed", "time_s", "mean_speed", "rms_accel", "action_change", "boundary_hits"]
ORDER = ["expert", "bead_dagger", "bead_ppo", "bead_dagger_ppo", "bead_published",
         "pp_fixed_original", "pp_fixed_tuned", "pp_adaptive_original", "pp_adaptive_tuned",
         "rlpp_joint", "rlpp_ld_only", "rlpp_published",
         "pp_fixed_tuned_matched", "pp_adaptive_tuned_matched", "rlpp_joint_matched"]
PLANNED = [("bead_dagger_ppo", "bead_dagger"), ("bead_ppo", "bead_dagger_ppo"), ("bead_dagger_ppo", "expert"),
           ("bead_dagger_ppo", "pp_fixed_tuned"), ("bead_dagger_ppo", "pp_fixed_tuned_matched"),
           ("rlpp_joint", "rlpp_ld_only"), ("rlpp_joint", "pp_fixed_tuned")]
PRIMARY = ["rmse", "geom010"]


def load():
    df = pd.read_csv(FINAL / "episodes.csv")
    df["completed"] = df["completed"].astype(str).str.lower().eq("true").astype(float)
    return df


def per_ref_seed(df):
    """Mean over evaluation episodes -> one value per (controller, train_seed, ref)."""
    return df.groupby(["controller", "train_seed", "family", "ref"])[METRICS].mean(numeric_only=True).reset_index()


def summary(prs, family):
    d = prs[prs.family == family]
    per_seed = d.groupby(["controller", "train_seed"])[METRICS].mean().reset_index()
    out = []
    for c in ORDER:
        s = per_seed[per_seed.controller == c]
        if s.empty:
            continue
        row = {"controller": c, "n_train_seeds": int(len(s))}
        for m in METRICS:
            row[m] = float(s[m].mean())
            row[m + "_sd"] = float(s[m].std(ddof=1)) if len(s) > 1 else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def per_circuit(prs):
    d = prs[prs.family == "circuit"]
    return d.groupby(["controller", "ref"])[METRICS].agg(["mean", "std"]).reset_index()


def wilcoxon_exact(diff):
    """Exact two-sided Wilcoxon signed-rank p-value (zero differences dropped)."""
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_plus = ranks[d > 0].sum()
    total = ranks.sum()
    # enumerate all 2^n sign patterns (n <= 20 here)
    signs = np.array(list(product([0, 1], repeat=n)))
    dist = signs @ ranks
    stat = min(w_plus, total - w_plus)
    p = np.mean((dist <= stat + 1e-9) | (dist >= total - stat - 1e-9))
    return float(min(1.0, p)), float(w_plus)


def bootstrap_ci(prs, a, b, metric, family="circuit", reps=10000, rng=np.random.default_rng(0)):
    d = prs[prs.family == family]
    refs = sorted(d.ref.unique())
    A = d[d.controller == a].pivot_table(index="ref", columns="train_seed", values=metric).reindex(refs)
    B = d[d.controller == b].pivot_table(index="ref", columns="train_seed", values=metric).reindex(refs)
    Av, Bv = A.to_numpy(), B.to_numpy()
    diffs = []
    for _ in range(reps):
        ri = rng.integers(len(refs), size=len(refs))
        sa = rng.integers(Av.shape[1], size=Av.shape[1])
        sb = rng.integers(Bv.shape[1], size=Bv.shape[1])
        diffs.append(np.mean(Av[ri][:, sa].mean(1) - Bv[ri][:, sb].mean(1)))
    point = float(np.mean(Av.mean(1) - Bv.mean(1)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi), Av.mean(1) - Bv.mean(1)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def comparisons(prs, family="circuit"):
    rows = []
    for a, b in PLANNED:
        if not ({a, b} <= set(prs.controller)):
            continue
        for m in PRIMARY + ["time_s"]:
            point, lo, hi, per_ref = bootstrap_ci(prs, a, b, m, family)
            p, w = wilcoxon_exact(per_ref) if family == "circuit" else wilcoxon_normal(per_ref)
            rows.append({"A": a, "B": b, "metric": m, "mean_diff_A_minus_B": point, "ci95_lo": lo, "ci95_hi": hi,
                         "n_refs": int(len(per_ref)), "A_better_on": int(np.sum(per_ref < 0) if m != "geom010" else np.sum(per_ref > 0)),
                         "wilcoxon_p": p, "planned": m in PRIMARY})
    df = pd.DataFrame(rows)
    mask = df.planned.to_numpy()
    df["holm_p"] = np.nan
    df.loc[mask, "holm_p"] = holm(df.loc[mask, "wilcoxon_p"].to_numpy())
    return df


def wilcoxon_normal(diff):
    from scipy.stats import wilcoxon
    d = np.asarray(diff, float)
    if np.allclose(d, 0):
        return 1.0, 0.0
    r = wilcoxon(d, zero_method="wilcox", method="approx")
    return float(r.pvalue), float(r.statistic)


def main():
    df = load()
    prs = per_ref_seed(df)
    out = FINAL / "tables"
    out.mkdir(exist_ok=True)
    res = {}
    for fam in ("circuit", "closed", "curve"):
        s = summary(prs, fam)
        s.to_csv(out / f"summary_{fam}.csv", index=False)
        res[fam] = s
    per_circuit(prs).to_csv(out / "per_circuit.csv")
    prs[prs.family == "circuit"].groupby(["controller", "train_seed"])[METRICS].mean().to_csv(out / "per_seed_circuits.csv")
    comp = comparisons(prs, "circuit")
    comp.to_csv(out / "comparisons_circuits.csv", index=False)
    comp_s = pd.concat([comparisons(prs, f).assign(family=f) for f in ("closed", "curve")])
    comp_s.to_csv(out / "comparisons_synthetic.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["controller", "n_train_seeds", "geom010", "geom010_sd", "rmse", "rmse_sd", "p95", "max_err",
            "ordered_cov", "completed", "time_s", "mean_speed"]
    for fam, s in res.items():
        print(f"\n=== {fam} ===")
        print(s[cols].round(4).to_string(index=False))
    print("\n=== planned comparisons (test circuits) ===")
    print(comp.round(4).to_string(index=False))
    meta = {"n_episodes": int(len(df)), "controllers": sorted(df.controller.unique()),
            "refs_per_family": df.groupby("family").ref.nunique().to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
