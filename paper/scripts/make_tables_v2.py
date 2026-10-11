"""Writes every numeric table of the revised manuscript (paper/revised/tables/*.tex) directly from
results/final/episodes.csv, results/final/tables/comparisons_circuits.csv, runs/pp_tuning.json and
datasets/split_manifest.json, so that no number is transcribed by hand.

Usage: python paper/scripts/make_tables_v2.py   (after experiments/analyze.py)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "final"
import os
OUT = Path(os.environ.get("TABLES_OUT", ROOT / "paper" / "revised" / "tables"))   # override to regenerate elsewhere
OUT.mkdir(parents=True, exist_ok=True)
DF = pd.read_csv(FINAL / "episodes.csv")
DF["completed"] = DF["completed"].astype(str).str.lower().eq("true").astype(float)

NAME = {"expert": "Geometric expert (4)", "bead_dagger": "Bead-DAgger", "bead_dagger_ppo": "Bead-DAgger$\\to$PPO",
        "bead_ppo": "Bead-PPO (scratch)", "bead_published": "Published Bead ckpt.$^\\ast$",
        "pp_fixed_original": "Fixed PP (orig.\\ gains)", "pp_fixed_tuned": "Fixed PP (tuned)",
        "pp_adaptive_original": "Adaptive PP (orig.)", "pp_adaptive_tuned": "Adaptive PP (tuned)",
        "rlpp_joint": "RL--PP (joint $L_d,g$) [4]", "rlpp_ld_only": "RL--PP ($L_d$ only)",
        "rlpp_published": "Published RL--PP ckpt.$^\\dagger$",
        "pp_fixed_tuned_matched": "Fixed PP (tuned)", "pp_adaptive_tuned_matched": "Adaptive PP (tuned)",
        "rlpp_joint_matched": "RL--PP (joint $L_d,g$)"}
BEAD = ["expert", "bead_dagger", "bead_dagger_ppo", "bead_ppo", "bead_published"]
PP_A = ["pp_fixed_original", "pp_fixed_tuned", "pp_adaptive_original", "pp_adaptive_tuned",
        "rlpp_joint", "rlpp_ld_only", "rlpp_published"]
PP_B = ["pp_fixed_tuned_matched", "pp_adaptive_tuned_matched", "rlpp_joint_matched"]


def seed_level(family):
    d = DF[DF.family == family]
    per = d.groupby(["controller", "train_seed", "ref"]).mean(numeric_only=True).reset_index()
    return per, per.groupby(["controller", "train_seed"]).mean(numeric_only=True).reset_index()


def fmt(v, nd):
    return f"{v:.{nd}f}"


def cell(s, m, scale, nd):
    v = s[m].to_numpy() * scale
    if len(v) > 1 and m in ("geom010", "rmse"):   # seed range shown for the primary metrics only
        return f"{fmt(v.mean(), nd)}\\,[{fmt(v.min(), nd)}, {fmt(v.max(), nd)}]"
    return fmt(v.mean(), nd)


RANKED = {"geom005": max, "geom010": max, "rmse": min, "p95": min, "max_err": min, "ordered_cov": max}


def bold_best(rows, cols):
    """rows: list of (name, {metric: (mean, text)}); bold the best displayed mean per ranked column."""
    for m, _, nd in cols:
        if m not in RANKED:
            continue
        vals = [round(r[1][m][0], nd) for r in rows]
        best = RANKED[m](vals)
        for r, v in zip(rows, vals):
            if v == best:
                mean, text = r[1][m]
                head, sep, tail = text.partition("\\,")
                r[1][m] = (mean, f"\\textbf{{{head}}}{sep}{tail}")
    return rows


def circuits_table():
    _, sm = seed_level("circuit")
    cols = [("geom005", 100, 1), ("geom010", 100, 1), ("rmse", 100, 2), ("p95", 100, 2), ("max_err", 100, 2),
            ("ordered_cov", 100, 1), ("completed", 100, 0), ("time_s", 1, 1), ("mean_speed", 1, 2)]
    blocks = [("Bead plant (holonomic point mass, damping-limited)", BEAD),
              ("Bicycle plant, native speed profile (Track A)", PP_A),
              ("Bicycle plant, speed capped at $v_{\\mathrm{match}}=1.256$ m/s (Track B)", PP_B)]
    rows = []
    for _, names in blocks:
        for c in names:
            s = sm[sm.controller == c]
            rows.append((c, {m: (float(s[m].mean() * k), cell(s, m, k, nd)) for m, k, nd in cols}))
    bold_best(rows, cols)          # best over all 15 controllers, as in the earlier Table III
    by = dict(rows)
    lines = []
    for i, (title, names) in enumerate(blocks):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for c in names:
            lines.append(NAME[c] + " & " + " & ".join(by[c][m][1] for m, _, _ in cols) + "\\\\")
    (OUT / "tab_circuits.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paper_circuits_table():
    per, _ = seed_level("circuit")
    refs = ["circuit_Hockenheim", "circuit_Montreal", "circuit_YasMarina"]
    names = ["expert", "bead_dagger", "bead_dagger_ppo", "bead_published", "pp_fixed_original", "pp_fixed_tuned",
             "rlpp_joint", "rlpp_published"]
    cols = []
    rows = [(c, {}) for c in names]
    for r in refs:
        for m, nd in (("geom010", 1), ("rmse", 2)):
            key = f"{r}:{m}"
            cols.append((key, 100, nd))
            RANKED[key] = RANKED[m]
            for c, d in rows:
                v = 100 * per[(per.controller == c) & (per.ref == r)][m].mean()
                d[key] = (float(v), f"{v:.{nd}f}")
    bold_best(rows, cols)
    lines = [NAME[c] + " & " + " & ".join(d[k][1] for k, _, _ in cols) + "\\\\" for c, d in rows]
    (OUT / "tab_paper_circuits.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def shapes_table():
    names = ["expert", "bead_dagger", "bead_dagger_ppo", "bead_ppo", "pp_fixed_tuned", "rlpp_joint", "bead_published",
             "rlpp_published"]
    cols = [("geom010", 100, 1), ("rmse", 100, 2), ("ordered_cov", 100, 1), ("time_s", 1, 1)]
    lines = []
    for fam, title in (("closed", "Closed shapes (50)"), ("curve", "Open curves (50)")):
        per, _ = seed_level(fam)
        rows = []
        for c in names:
            b = per[per.controller == c].groupby("ref").mean(numeric_only=True)
            g, sd = 100 * b.geom010.mean(), 100 * b.geom010.std(ddof=1)
            rows.append((c, {"geom010": (g, f"{g:.1f}\\, $\\pm$ {sd:.1f}"),
                             "rmse": (100 * b.rmse.mean(), f"{100 * b.rmse.mean():.2f}"),
                             "ordered_cov": (100 * b.ordered_cov.mean(), f"{100 * b.ordered_cov.mean():.1f}"),
                             "time_s": (b.time_s.mean(), f"{b.time_s.mean():.1f}")}))
        bold_best(rows, cols)
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}}\\\\")
        lines += [NAME[c] + " & " + " & ".join(d[m][1] for m, _, _ in cols) + "\\\\" for c, d in rows]
    (OUT / "tab_shapes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparisons_table():
    c = pd.read_csv(FINAL / "tables" / "comparisons_circuits.csv")
    c = c[c.metric.isin(["rmse", "geom010"])]
    short = {"bead_dagger_ppo": "DAgger$\\to$PPO", "bead_dagger": "DAgger", "bead_ppo": "PPO (scratch)",
             "expert": "Expert", "pp_fixed_tuned": "Fixed PP", "pp_fixed_tuned_matched": "Fixed PP @$v_{\\mathrm{match}}$",
             "rlpp_joint": "RL--PP joint", "rlpp_ld_only": "RL--PP $L_d$"}
    lines = []
    for _, r in c.iterrows():
        sc, unit = (100, "cm") if r.metric == "rmse" else (100, "pp")
        metric = "RMSE" if r.metric == "rmse" else "Geom@0.10"
        lines.append(f"{short[r.A]} vs {short[r.B]} & {metric} & {sc * r.mean_diff_A_minus_B:+.2f} {unit} & "
                     f"[{sc * r.ci95_lo:+.2f}, {sc * r.ci95_hi:+.2f}] & {int(r.A_better_on)}/8 & "
                     f"{r.wilcoxon_p:.4f} & {r.holm_p:.3f}\\\\")
    (OUT / "tab_comparisons.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def numbers():
    """Scalars quoted in the text, written as LaTeX macros."""
    m = json.loads((ROOT / "datasets" / "split_manifest.json").read_text())
    ext = [e["extraction_error_m"] for e in m["entries"] if e["family"] == "f1tenth_circuit"]
    tune = json.loads((ROOT / "runs" / "pp_tuning.json").read_text())
    fs = json.loads((ROOT / "paper" / "revised" / "figures" / "fig7_stats.json").read_text())
    _, sm = seed_level("circuit")
    g = lambda c, k: float(sm[sm.controller == c][k].mean())  # noqa: E731
    d = {
        "extMeanLo": min(e["gt_to_ref_mean"] for e in ext) * 100, "extMeanHi": max(e["gt_to_ref_mean"] for e in ext) * 100,
        "extMax": max(e["gt_to_ref_max"] for e in ext) * 100,
        "fixL": tune["fixed_native"]["best"]["params"][0], "fixG": tune["fixed_native"]["best"]["params"][1],
        "fixLm": tune["fixed_matched"]["best"]["params"][0], "fixGm": tune["fixed_matched"]["best"]["params"][1],
        "adA": tune["adaptive_native"]["best"]["params"][0], "adB": tune["adaptive_native"]["best"]["params"][1],
        "rlppLmin": fs["L_min"], "rlppLmax": fs["L_max"], "rlppGmin": fs["g_min"], "rlppGmax": fs["g_max"],
        "rlppGstd": fs["g_std"], "rLk": fs["r_L_kmax"], "rGk": fs["r_g_kmax"],
        "rmseExpert": g("expert", "rmse") * 100, "rmseDagger": g("bead_dagger", "rmse") * 100,
        "rmseDP": g("bead_dagger_ppo", "rmse") * 100, "rmsePPO": g("bead_ppo", "rmse") * 100,
        "rmseFix": g("pp_fixed_tuned", "rmse") * 100, "rmseAd": g("pp_adaptive_tuned", "rmse") * 100,
        "rmseRL": g("rlpp_joint", "rmse") * 100, "rmsePubBead": g("bead_published", "rmse") * 100,
        "rmsePubRL": g("rlpp_published", "rmse") * 100, "rmseFixOrig": g("pp_fixed_original", "rmse") * 100,
        "rmseFixM": g("pp_fixed_tuned_matched", "rmse") * 100, "rmseAdM": g("pp_adaptive_tuned_matched", "rmse") * 100,
        "rmseRLM": g("rlpp_joint_matched", "rmse") * 100,
        "tExpert": g("expert", "time_s"), "tFix": g("pp_fixed_tuned", "time_s"), "tDP": g("bead_dagger_ppo", "time_s"),
        "tPPO": g("bead_ppo", "time_s"), "compPPO": g("bead_ppo", "completed") * 100,
        "gPPO": g("bead_ppo", "geom010") * 100,
        "nEpisodes": len(DF),
    }
    bead = DF[(DF.family == "circuit") & (DF.plant == "bead")].groupby("controller").env_reward.mean()
    d.update({"rewExpert": bead["expert"], "rewDagger": bead["bead_dagger"], "rewDP": bead["bead_dagger_ppo"],
              "rewPPO": bead["bead_ppo"]})
    # Validation-only PPO diagnostic (experiments/ppo_diagnostic.py), averaged over the two seeds.
    diag = json.loads((ROOT / "results/final/verification/ppo_diagnostic.json").read_text())
    for key, tag in (("dagger_det", "DagDet"), ("dagger_stoch", "DagSto"), ("dagger_ppo_det", "DPDet"),
                     ("dagger_ppo_stoch", "DPSto")):
        for q, short in (("reward", "rew"), ("rmse_cm", "rmse"), ("time_s", "t")):
            d[f"diag{tag}{short}"] = float(np.mean([diag[s][key][q] for s in ("seed0", "seed1")]))
    rep = json.loads((ROOT / "results/final/verification/report.json").read_text())
    g = rep["G_ppo_diagnostics"]
    kl_last = [v["approx_kl_first_last"][1] for v in g.values()]
    ev_last = [v["explained_variance_first_last"][1] for v in g.values()]
    frozen = [abs(v["approx_kl_while_frozen_max"]) for v in g.values() if v["approx_kl_while_frozen_max"] is not None]
    d.update({"klLo": min(kl_last), "klHi": max(kl_last), "evLo": min(ev_last), "evHi": max(ev_last)})
    frozen_exp = int(np.floor(np.log10(max(frozen))))
    acc = rep["A_accounting"]
    d.update({"nRefs": acc["references_in_manifest"], "nInst": acc["controller_instances"],
              "nSynthEval": acc["test_synthetic_evaluated"]})
    lines = [f"\\newcommand{{\\VfrozenKLexp}}{{{frozen_exp + 1}}}"]   # |KL| while frozen < 10^(this)
    for k, v in d.items():
        if k in ("klLo", "klHi"):
            lines.append(f"\\newcommand{{\\V{k}}}{{{v:.3f}}}")
            continue
        if k.startswith("diag") and k.endswith(("rew", "t")):
            lines.append(f"\\newcommand{{\\V{k}}}{{{v:.0f}}}")
            continue
        s = f"{v:.0f}" if k.startswith(("rew", "nEp", "comp", "nRefs", "nInst", "nSynth")) else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"\\newcommand{{\\V{k}}}{{{s}}}")  # V prefix avoids clashes with class internals
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    circuits_table(); paper_circuits_table(); shapes_table(); comparisons_table(); numbers()
    print("tables ->", OUT)
