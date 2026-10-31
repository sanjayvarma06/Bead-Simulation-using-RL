"""Writes every numeric table of the revised manuscript (paper/revised/tables/*.tex) and the scalars
quoted in the text (numbers.tex) directly from the result files, so that no number is transcribed by
hand: results/final/episodes.csv, results/final/tables/comparisons_circuits.csv,
results/final/expert_k/episodes.csv, results/final/verification/*.json (incl. car_lateral_accel.json), runs/pp_tuning.json,
runs/expert_k_tuning.json, runs/bead/*/seed*/val_log.csv and datasets/split_manifest.json.

Usage: python paper/scripts/make_tables_v2.py   (after experiments/analyze.py, ppo_diagnostic.py,
       selection_sensitivity.py, verify_results.py and make_figures_v2.py)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
FINAL = ROOT / "results" / "final"
RUNS = ROOT / "runs"
OUT = Path(os.environ.get("TABLES_OUT", ROOT / "paper" / "revised" / "tables"))   # override to regenerate elsewhere
OUT.mkdir(parents=True, exist_ok=True)
DF = pd.read_csv(FINAL / "episodes.csv")
DF["completed"] = DF["completed"].astype(str).str.lower().eq("true").astype(float)

# "(\ref{eq:expert})" prints the expert's equation number of whichever manuscript includes the table.
NAME = {"expert": "Geometric expert (\\ref{eq:expert})", "bead_pp": "Bead-PP (PP law on bead)$^\\S$",
        "bead_dagger": "Bead-DAgger", "bead_dagger_bpp": "Bead-DAgger, Bead-PP teacher",
        "bead_dagger_ppo": "Bead-DAgger$\\to$PPO",
        "bead_dagger_ppo_bc10": "Bead-DAgger$\\to$PPO, $\\lambda=10$ (3 seeds)",
        "bead_dagger_ppo_acc": "Bead-DAgger$\\to$PPO, prog.\\ rew.",
        "bead_ppo": "Bead-PPO (scratch)", "bead_ppo_acc": "Bead-PPO (scratch), prog.\\ rew.",
        "bead_published": "Bead, naive protocol$^\\ast$",
        "pp_fixed_original": "Fixed PP (hand-set gains)", "pp_fixed_tuned": "Fixed PP (tuned)",
        "pp_adaptive_original": "Adaptive PP (hand-set)", "pp_adaptive_tuned": "Adaptive PP (tuned)",
        "rlpp_joint": "RL--PP (joint $L_d,g$) [4]", "rlpp_ld_only": "RL--PP ($L_d$ only)",
        "rlpp_published": "RL--PP, naive protocol$^\\dagger$",
        "pp_fixed_tuned_matched": "Fixed PP (tuned)", "pp_adaptive_tuned_matched": "Adaptive PP (tuned)",
        "rlpp_joint_matched": "RL--PP (joint $L_d,g$)",
        "stanley_tuned": "Stanley (tuned)", "stanley_tuned_matched": "Stanley (tuned)"}
BEAD = ["expert", "bead_pp", "bead_dagger", "bead_dagger_bpp", "bead_dagger_ppo", "bead_dagger_ppo_bc10", "bead_dagger_ppo_acc", "bead_ppo", "bead_ppo_acc",
        "bead_published"]
PP_A = ["pp_fixed_original", "pp_fixed_tuned", "pp_adaptive_original", "pp_adaptive_tuned", "stanley_tuned",
        "rlpp_joint", "rlpp_ld_only", "rlpp_published"]
PP_B = ["pp_fixed_tuned_matched", "pp_adaptive_tuned_matched", "stanley_tuned_matched", "rlpp_joint_matched"]
SHORT = {"bead_dagger_ppo": "DAgger$\\to$PPO", "bead_dagger_ppo_acc": "DAgger$\\to$PPO (prog.)",
         "bead_dagger": "DAgger", "bead_dagger_bpp": "DAgger (Bead-PP teacher)", "bead_ppo": "PPO (scratch)",
         "bead_dagger_ppo_bc10": "DAgger$\\to$PPO ($\\lambda=10$)", "bead_ppo_acc": "PPO (scratch, prog.)",
         "expert": "Expert", "bead_pp": "Bead-PP", "pp_fixed_tuned": "Fixed PP",
         "pp_fixed_tuned_matched": "Fixed PP @$v_{\\mathrm{match}}$", "pp_adaptive_tuned": "Adaptive PP",
         "rlpp_joint": "RL--PP joint", "rlpp_ld_only": "RL--PP $L_d$",
         "stanley_tuned": "Stanley", "stanley_tuned_matched": "Stanley @$v_{\\mathrm{match}}$"}


def seed_level(family, df=DF):
    d = df[df.family == family]
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
    bold_best(rows, cols)          # best over all controllers
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
    names = ["expert", "bead_dagger", "bead_published", "pp_fixed_original", "pp_fixed_tuned",
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
    names = ["expert", "bead_pp", "bead_dagger", "bead_dagger_ppo", "bead_ppo", "pp_fixed_tuned", "rlpp_joint",
             "bead_published", "rlpp_published"]
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
    c = c[c.metric == "rmse"]
    # Same paired comparison on the RMSE against the true centerlines, recomputed from the stored
    # trajectories by experiments/centerline_metrics.py (same bootstrap over circuits and seeds).
    tc = pd.read_csv(CL / "comparisons.csv").set_index(["A", "B"])
    lines = []
    for title, part in (("Primary (Holm over these four)", c[c.primary]), ("Exploratory (no correction)", c[~c.primary])):
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for _, r in part.iterrows():
            holm = f"{r.holm_p:.3f}" if r.primary else "--"
            sg = lambda v: f"{100 * v:+.2f}".replace("-", "$-$")  # noqa: E731
            if (r.A, r.B) in tc.index:
                t = tc.loc[(r.A, r.B)]
                true = f"{sg(t.mean_diff_A_minus_B)} & [{sg(t.ci95_lo)}, {sg(t.ci95_hi)}]"
            else:
                true = "-- & --"
            lines.append(f"{SHORT[r.A]} vs {SHORT[r.B]} & {sg(r.mean_diff_A_minus_B)} & "
                         f"[{sg(r.ci95_lo)}, {sg(r.ci95_hi)}] & {int(r.A_better_on)}/{int(r.n_refs)} & "
                         f"{r.wilcoxon_p:.4f} & {holm} & {true}\\\\")
        if title.startswith("Primary"):
            lines.append("\\midrule")
    (OUT / "tab_comparisons.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


CL = FINAL / "centerline"
TRUE_ROWS = ["expert", "bead_pp", "bead_dagger", "bead_dagger_bpp", "bead_dagger_ppo", "bead_ppo",
             "pp_fixed_tuned", "pp_fixed_tuned_matched", "pp_adaptive_tuned", "stanley_tuned", "stanley_tuned_matched",
             "rlpp_joint"]


def true_centerline_table():
    """Extracted-reference RMSE next to RMSE / P95 / Geom@0.10 against the true centerline (test circuits)."""
    _, sm = seed_level("circuit")
    tr = pd.read_csv(CL / "summary.csv", index_col=0)
    cols = [("rmse", 2), ("rmse_true", 2), ("p95_true", 2), ("geom010_true", 1)]
    rows = []
    for c in TRUE_ROWS:
        if c not in tr.index:
            continue
        v = {"rmse": 100 * float(sm[sm.controller == c].rmse.mean()), "rmse_true": 100 * tr.loc[c, "rmse_true"],
             "p95_true": 100 * tr.loc[c, "p95_true"], "geom010_true": 100 * tr.loc[c, "geom010_true"]}
        rows.append((c, {m: (v[m], f"{v[m]:.{nd}f}") for m, nd in cols}))
    RANKED.update({"rmse_true": min, "p95_true": min, "geom010_true": max})
    bold_best(rows, [(m, 100, nd) for m, nd in cols])
    label = dict(NAME, pp_fixed_tuned="Fixed PP (tuned)", pp_fixed_tuned_matched="Fixed PP (tuned) @$v_{\\mathrm{match}}$",
                 stanley_tuned_matched="Stanley (tuned) @$v_{\\mathrm{match}}$",
                 rlpp_joint="RL--PP (joint $L_d,g$)", bead_pp="Bead-PP (PP law on bead)")
    lines = [label[c] + " & " + " & ".join(d[m][1] for m, _ in cols) + "\\\\" for c, d in rows]
    (OUT / "tab_truecl.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dynamic_table(ds):
    """Controllers at v_match on the dynamic single-track car (a) tyres only, (b) plus rate limit."""
    def settings(job):
        p = json.loads((RUNS / "extra_tuning" / f"{job}.json").read_text())["best"]["params"]
        return f"({p[0]:g}, {p[1]:g})"
    rows = [("Fixed PP, re-tuned", "fixed_retuned", "fixed"), ("Fixed PP, re-tuned, $g>1.4$", "fixed_retuned_highg", "fixedhigh"),
            ("Stanley, re-tuned", "stanley_retuned", "stanley"), ("RL--PP, retrained", "rlpp_retrained", None),
            ("Fixed PP, unchanged", "fixed_transfer", None), ("Stanley, unchanged", "stanley_transfer", None),
            ("RL--PP, unchanged", "rlpp_transfer", None)]
    lines = []
    for i, (label, c, job) in enumerate(rows):
        if i == 4:
            lines.append("\\midrule")
        cells = [label]
        for exp, tag in (("dynamic_dt", "dt"), ("dynamic_dr", "dr")):
            r = ds.loc[(exp, c)]
            cells += [settings(f"{job}_{tag}") if job else "--", f"{r.rmse_cm:.2f}", f"{r.rmse_true_cm:.2f}", f"{r.completed_pct:.0f}"]
        lines.append(" & ".join(cells) + "\\\\")
    (OUT / "tab_dynamic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def spacing_table():
    from common import load_path
    """Waypoint spacing of every extracted circuit reference (supplementary)."""
    sp = pd.read_csv(CL / "spacing.csv")
    m = json.loads((ROOT / "datasets" / "split_manifest.json").read_text())
    split = {e["id"]: e["split"] for e in m["entries"]}
    sp["split"] = sp.ref.map(split)
    sp = sp.sort_values(["split", "ref"], key=lambda s: s.map({"train": 0, "val": 1, "test": 2}) if s.name == "split" else s)
    # A loop whose extracted polyline does not close exactly ends with one short segment (the
    # final skeleton pixels hook back towards the start). That segment is marked and explained;
    # it is not a junction between pieces (all circuit references have zero breaks).
    lines, notes = [], []
    for r in sp.itertuples():
        pts, brk = load_path(r.ref)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        closing = int(np.argmin(d)) == len(d) - 1 and d[-1] < 0.8 * np.median(d)
        mark = "$^\\ddagger$" if closing else ""
        lines.append(f"{r.ref.replace('circuit_', '')} & {r.split} & {r.n_points} & {100 * r.spacing_median_m:.1f} & "
                     f"{100 * r.spacing_min_m:.1f}{mark}--{100 * r.spacing_max_m:.1f}\\\\")
        if closing:
            assert not brk
            notes.append(f"{r.ref.replace('circuit_', '')} {100 * d[-1]:.1f}~cm, loop open by "
                         f"{100 * np.linalg.norm(pts[-1] - pts[0]):.1f}~cm, minimum of the other segments "
                         f"{100 * d[:-1].min():.1f}~cm")
    (OUT / "tab_spacing.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # compact version: two circuits per row (left column first half, right column second half)
    cells = [ln[:-2] for ln in lines]
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:] + [" & & & & "] * (2 * half - len(cells))
    (OUT / "tab_spacing_2col.tex").write_text(
        "\n".join(f"{a} & {b}\\\\" for a, b in zip(left, right)) + "\n", encoding="utf-8")
    note = ("$^\\ddagger$Shortest segment is the last one, where the extracted loop stops short of its start "
            "and its final points hook back slightly; it is not a junction between pieces (no circuit "
            "reference has a break): " + "; ".join(notes) + ".") if notes else ""
    (OUT / "spacing_note.tex").write_text(note + "\n", encoding="utf-8")


def track_scale():
    from prepare_data import centerline_in_arena, CIRCUIT_DIR
    m = json.loads((ROOT / "datasets" / "split_manifest.json").read_text())
    ratios = []
    for e in m["entries"]:
        if e["family"] != "f1tenth_circuit":
            continue
        csv = CIRCUIT_DIR / "centerlines" / f"{e['id'].replace('circuit_', '')}_centerline.csv"
        src = pd.read_csv(csv, comment="#", header=None).iloc[:, :2].values.astype(float)
        src = np.vstack([src, src[:1]])
        ar = centerline_in_arena(csv)
        ratios.append(np.linalg.norm(np.diff(src, axis=0), axis=1).sum() / np.linalg.norm(np.diff(ar, axis=0), axis=1).sum())
    return min(ratios), float(np.median(ratios)), max(ratios)


def numbers():
    """Scalars quoted in the text, written as LaTeX macros."""
    m = json.loads((ROOT / "datasets" / "split_manifest.json").read_text())
    ext = [e["extraction_error_m"] for e in m["entries"] if e["family"] == "f1tenth_circuit"]
    tune = json.loads((RUNS / "pp_tuning.json").read_text())
    fs = json.loads((ROOT / "paper" / "revised" / "figures" / "fig7_stats.json").read_text())
    _, sm = seed_level("circuit")
    g = lambda c, k: float(sm[sm.controller == c][k].mean())  # noqa: E731
    P = lambda key, i: tune[key]["best"]["params"][i]  # noqa: E731
    fixed_grid = tune["fixed_native"]["grid"]
    def best_at_L(L):
        return 100 * min(r["rmse_circuits"] for r in fixed_grid if abs(r["params"][0] - L) < 1e-9)
    d = {
        "extMeanLo": min(e["gt_to_ref_mean"] for e in ext) * 100, "extMeanHi": max(e["gt_to_ref_mean"] for e in ext) * 100,
        "extMax": max(e["gt_to_ref_max"] for e in ext) * 100,
        "fixL": P("fixed_native", 0), "fixG": P("fixed_native", 1), "fixLm": P("fixed_matched", 0),
        "fixGm": P("fixed_matched", 1), "adA": P("adaptive_native", 0), "adB": P("adaptive_native", 1),
        "adAm": P("adaptive_matched", 0), "adBm": P("adaptive_matched", 1),
        "bppL": P("beadpp_matched", 0), "bppG": P("beadpp_matched", 1),
        "tuneRa": best_at_L(0.1), "tuneRb": best_at_L(0.125), "tuneRc": best_at_L(0.15),
        "rlppLmin": fs["L_min"], "rlppLmax": fs["L_max"], "rlppGmin": fs["g_min"], "rlppGmax": fs["g_max"],
        "rlppGstd": fs["g_std"], "rLk": fs["r_L_kmax"], "rGk": fs["r_g_kmax"],
        "rlppLmed": fs["all_L_median"], "rlppLlo": fs["all_L_p05"], "rlppLhi": fs["all_L_p95"],
        "rlppLband": 100 * fs["all_L_frac_teacher_band"],
        "tExpert": g("expert", "time_s"), "tFix": g("pp_fixed_tuned", "time_s"),
        "tFixM": g("pp_fixed_tuned_matched", "time_s"), "tBpp": g("bead_pp", "time_s"),
        "compPPO": g("bead_ppo", "completed") * 100, "compPPOacc": g("bead_ppo_acc", "completed") * 100,
        "gPPO": g("bead_ppo", "geom010") * 100, "gPPOacc": g("bead_ppo_acc", "geom010") * 100,
        "nEpisodes": len(DF),
    }
    for c, key in (("expert", "Expert"), ("bead_dagger", "Dagger"), ("bead_dagger_ppo", "DP"),
                   ("bead_dagger_ppo_acc", "DPacc"), ("bead_ppo", "PPO"), ("bead_ppo_acc", "PPOacc"),
                   ("bead_pp", "Bpp"), ("pp_fixed_tuned", "Fix"), ("pp_adaptive_tuned", "Ad"), ("rlpp_joint", "RL"),
                   ("rlpp_ld_only", "RLld"), ("bead_published", "PubBead"), ("rlpp_published", "PubRL"),
                   ("pp_fixed_original", "FixOrig"), ("pp_adaptive_original", "AdOrig"),
                   ("pp_fixed_tuned_matched", "FixM"), ("pp_adaptive_tuned_matched", "AdM"),
                   ("rlpp_joint_matched", "RLM")):
        d[f"rmse{key}"] = g(c, "rmse") * 100
    bead = DF[(DF.family == "circuit") & (DF.plant == "bead")].groupby("controller").env_reward.mean()
    d.update({"rewExpert": bead["expert"], "rewDagger": bead["bead_dagger"], "rewDP": bead["bead_dagger_ppo"],
              "rewPPO": bead["bead_ppo"]})
    # validation-only diagnostics (experiments/ppo_diagnostic.py)
    diag = json.loads((FINAL / "verification" / "ppo_diagnostic.json").read_text())
    ex = diag["expert"]
    hi = max(ex, key=float)
    for mode, mk in (("base", "B"), ("progress", "P")):
        d[f"exRew{mk}zero"], d[f"exRew{mk}noise"] = ex["0.000"][mode]["reward"], ex[hi][mode]["reward"]
    d.update({"exRmseZero": ex["0.000"]["base"]["rmse_cm"], "exRmseNoise": ex[hi]["base"]["rmse_cm"],
              "exTZero": ex["0.000"]["base"]["time_s"], "exTNoise": ex[hi]["base"]["time_s"], "exStd": float(hi)})
    for v, vk in (("dagger_ppo", "DP"), ("dagger_ppo_acc", "DPacc")):
        rows = [r for k, r in diag["finetune"].items() if k.startswith(v + "/")]
        for key, tag in (("dagger_det", "DagDet"), ("dagger_stoch", "DagSto"), ("finetuned_det", "FtDet"),
                         ("finetuned_stoch", "FtSto")):
            for q, short in (("reward", "rew"), ("rmse_cm", "rmse")):
                d[f"diag{vk}{tag}{short}"] = float(np.mean([r[key][q] for r in rows]))
        d[f"diag{vk}StdEnd"] = float(np.mean([r["std_end"] for r in rows]))
    rep = json.loads((FINAL / "verification" / "report.json").read_text())
    gdiag = rep["G_ppo_diagnostics"]
    kl_last = [v["approx_kl_first_last"][1] for v in gdiag.values()]
    ev_last = [v["explained_variance_first_last"][1] for v in gdiag.values()]
    frozen = [abs(v["approx_kl_while_frozen_max"]) for v in gdiag.values() if v["approx_kl_while_frozen_max"] is not None]
    d.update({"klLo": min(kl_last), "klHi": max(kl_last), "evLo": min(ev_last), "evHi": max(ev_last)})
    kl_frozen_text = ("was exactly zero" if max(frozen) == 0.0
                      else f"stayed below $10^{{{int(np.floor(np.log10(max(frozen)))) + 1}}}$")
    acc = rep["A_accounting"]
    # counted from the episode records themselves, so they cannot lag behind a stale report
    n_inst = int(DF.groupby(["controller", "train_seed"]).ngroups)
    n_synth = int(DF[DF.family != "circuit"].ref.nunique())
    assert n_inst * (8 * 5 + n_synth) == len(DF), "episode accounting does not add up"
    assert acc["controller_instances"] == n_inst, "verification report is stale: rerun experiments/refresh_verification.py"
    d.update({"nRefs": acc["references_in_manifest"], "nInst": n_inst, "nSynthEval": n_synth})
    rr = rep["D_rerun"]
    assert all(v["same_length"] and v["max_abs_position_diff_m"] == 0.0 for v in rr.values()), "re-execution mismatch"
    d["nRerun"] = len(rr)
    # expert-k sweep
    ek = pd.read_csv(FINAL / "expert_k" / "episodes.csv")
    ek["completed"] = ek["completed"].astype(str).str.lower().eq("true").astype(float)
    ekc = ek[ek.family == "circuit"].groupby(["controller", "ref"]).mean(numeric_only=True).groupby("controller").mean()
    ksel = json.loads((RUNS / "expert_k_tuning.json").read_text())["selected"]
    for a, ak in (("counter", "C"), ("nearest", "N")):
        kbest = ksel[a]["kw"]["k"]
        d[f"ekSel{ak}"] = kbest
        d[f"ekRmse{ak}"] = 100 * ekc.loc[f"expert_k{kbest}_{a}", "rmse"]
        fail = [k for k in range(1, 11) if ekc.loc[f"expert_k{k}_{a}", "completed"] < 1.0]
        d[f"ekFail{ak}"] = min(fail) if fail else 11
        d[f"ekRmseThree{ak}"] = 100 * ekc.loc[f"expert_k3_{a}", "rmse"]
    # selection-score sensitivity
    ss = json.loads((FINAL / "verification" / "selection_sensitivity.json").read_text())
    d.update({"selSame": ss["n_same"], "selTotal": ss["n_total"], "selMaxDiff": ss["max_rmse_diff_cm"],
              "selMedDiff": ss["median_rmse_diff_cm_when_different"]})
    ad = ss["pp_grids"]["adaptive_native"]
    d.update({"adRmseSel": ad["rmse_cm_by_score"], "adRmseConst": ad["rmse_cm_by_rmse"]})
    # log_std_init pilot (seed 100, validation only)
    for tag, key in (("-1.0", "Pm"), ("-0.5", "Ph"), ("0.0", "Pz")):
        v = pd.read_csv(RUNS / "bead" / f"ppo_pilot_ls{tag}" / "seed100" / "val_log.csv").iloc[-1]
        d[f"pilot{key}rmse"] = 100 * float(v.rmse_circuits)
        d[f"pilot{key}score"] = float(v.score)
    # car lateral acceleration and track scale
    la = json.loads((FINAL / "verification" / "car_lateral_accel.json").read_text())
    d.update({"latAccP": la["fixed_native"]["p99"], "latAccFrac": 100 * la["fixed_native"]["frac_above_10"],
              "latAccMaxM": la["fixed_matched"]["max"]})
    d["scaleLo"], d["scaleMed"], d["scaleHi"] = track_scale()
    # paired comparisons quoted in the text (extracted reference; RMSE in cm)
    comp = pd.read_csv(FINAL / "tables" / "comparisons_circuits.csv")
    comp = comp[comp.metric == "rmse"].set_index(["A", "B"])
    def put(prefix, a, b, table=comp):
        r = table.loc[(a, b)]
        d[f"{prefix}Diff"], d[f"{prefix}Lo"], d[f"{prefix}Hi"] = (100 * r.mean_diff_A_minus_B, 100 * r.ci95_lo, 100 * r.ci95_hi)
        d[f"{prefix}Better"], d[f"{prefix}P"] = int(r.A_better_on), float(r.wilcoxon_p)
    put("cmpPone", "bead_dagger", "pp_fixed_tuned_matched")
    put("cmpPfour", "bead_pp", "expert")
    put("cmpRLJL", "rlpp_joint", "rlpp_ld_only")
    if ("bead_dagger_bpp", "bead_dagger") in comp.index:
        put("cmpBppDag", "bead_dagger_bpp", "bead_dagger")
        put("cmpBppTeach", "bead_dagger_bpp", "bead_pp")
        d["rmseDaggerBpp"] = g("bead_dagger_bpp", "rmse") * 100
    # behaviour-cloning weight check (experiments/bc_check.py; validation only) and its test result
    bcj = json.loads((FINAL / "verification" / "bc_check.json").read_text())
    for lam, key in (("0.1", "Low"), ("1", "One"), ("10", "High")):
        d[f"bcVal{key}"] = bcj["lambdas"][lam]["val_rmse_circuits_cm_mean"]
    d["bcValDagger"] = bcj["dagger_val_rmse_circuits_cm_mean"]
    d["bcSel"] = float(bcj["selected_lambda"])
    if ("bead_dagger_ppo_bc10", "bead_dagger") in comp.index:
        d["rmseDPbcHigh"] = g("bead_dagger_ppo_bc10", "rmse") * 100
        put("cmpBcDag", "bead_dagger_ppo_bc10", "bead_dagger")
        put("cmpBcDP", "bead_dagger_ppo_bc10", "bead_dagger_ppo")
    # true-centerline results (experiments/centerline_metrics.py)
    tr = pd.read_csv(CL / "summary.csv", index_col=0)
    for c, key in (("expert", "Expert"), ("bead_pp", "Bpp"), ("bead_dagger", "Dagger"), ("pp_fixed_tuned", "Fix"),
                   ("pp_fixed_tuned_matched", "FixM"), ("bead_dagger_ppo", "DP"), ("rlpp_joint", "RL"),
                   ("bead_dagger_bpp", "DaggerBpp")):
        if c in tr.index:
            d[f"rmseTrue{key}"] = 100 * float(tr.loc[c, "rmse_true"])
    trc = pd.read_csv(CL / "comparisons.csv").set_index(["A", "B"])
    trc["metric"] = "rmse_true"
    put("truePone", "bead_dagger", "pp_fixed_tuned_matched", trc)
    put("truePfour", "bead_pp", "expert", trc)
    if ("bead_dagger_bpp", "bead_pp") in trc.index:
        put("trueBppDag", "bead_dagger_bpp", "bead_dagger", trc)
        put("trueBppTeach", "bead_dagger_bpp", "bead_pp", trc)
        put("trueBppFixM", "bead_dagger_bpp", "pp_fixed_tuned_matched", trc)
        put("cmpBppFixM", "bead_dagger_bpp", "pp_fixed_tuned_matched")
    ex = sm.groupby("controller").rmse.mean()
    d["rankCorr"] = float(pd.Series(ex).rank().corr(tr.rmse_true.reindex(ex.index).rank()))
    d["nPrimTrue"] = int(sum(trc.loc[(a, b)].ci95_lo > 0 or trc.loc[(a, b)].ci95_hi < 0
                             for a, b in trc[trc.primary].index))
    # waypoint spacing of the extracted circuit references [cm]
    sp = pd.read_csv(CL / "spacing.csv")
    from common import split_ids as _split
    test = set(_split("test", "f1tenth_circuit"))
    d.update({"spMin": 100 * sp.spacing_median_m.min(), "spMax": 100 * sp.spacing_median_m.max(),
              "spMed": 100 * sp.spacing_median_m.median(),
              "spTestMin": 100 * sp[sp.ref.isin(test)].spacing_median_m.min(),
              "spTestMax": 100 * sp[sp.ref.isin(test)].spacing_median_m.max()})
    # naive-protocol contrast on Hockenheim / Montreal / Yas Marina: per-circuit RMSE ranges [cm]
    per, _ = seed_level("circuit")
    three = per[per.ref.isin(["circuit_Hockenheim", "circuit_Montreal", "circuit_YasMarina"])]
    for c, key in (("bead_published", "PubBead"), ("pp_fixed_original", "FixOrig"), ("rlpp_published", "PubRL"),
                   ("pp_fixed_tuned", "Fix"), ("expert", "Expert")):
        v = 100 * three[three.controller == c].groupby("ref").rmse.mean()
        d[f"three{key}Lo"], d[f"three{key}Hi"] = float(v.min()), float(v.max())
    rest = per[(~per.ref.isin(three.ref.unique())) & (per.controller == "bead_published")]
    d["restPubBead"] = 100 * float(rest.rmse.mean())
    d["threePubBeadMean"] = 100 * float(three[three.controller == "bead_published"].rmse.mean())
    # share of trajectory samples within 3 cm (first episode, training seed 0), as in the CDF figure
    from geometry_utils import points_to_polyline_distance
    from common import load_path, split_ids
    for c, s, key in (("expert", None, "Expert"), ("bead_pp", None, "Bpp"), ("bead_dagger", 0, "Dagger"),
                      ("bead_dagger_ppo", 0, "DP"), ("bead_ppo", 0, "PPO"), ("rlpp_joint", 0, "RL"),
                      ("pp_fixed_tuned", None, "Fix"), ("pp_adaptive_tuned", None, "Ad")):
        T = np.load(FINAL / "traj" / f"{c}__s{s}.npz")
        e = np.concatenate([points_to_polyline_distance(T[f"{r}__50000"], load_path(r)[0])
                            for r in split_ids("test", "f1tenth_circuit")])
        d[f"within{key}"] = 100 * float(np.mean(e <= 0.03))

    # Stanley baseline (experiments/tune_stanley.py, evaluate.py, front_axle_check.py)
    if (RUNS / "stanley_tuning.json").exists() and "stanley_tuned" in set(sm.controller):
        st = json.loads((RUNS / "stanley_tuning.json").read_text())
        d["stK"], d["stKs"] = st["stanley_native"]["best"]["params"]
        d["stKm"], d["stKsm"] = st["stanley_matched"]["best"]["params"]
        d["stNgrid"] = len(st["stanley_native"]["grid"])
        for c, key in (("stanley_tuned", "St"), ("stanley_tuned_matched", "StM")):
            d[f"rmse{key}"] = 100 * g(c, "rmse")
            d[f"comp{key}"] = 100 * g(c, "completed")
            d[f"rmseTrue{key}"] = 100 * float(tr.loc[c, "rmse_true"])
        put("cmpStFix", "stanley_tuned", "pp_fixed_tuned")
        put("cmpStFixM", "stanley_tuned_matched", "pp_fixed_tuned_matched")
        put("trueStFix", "stanley_tuned", "pp_fixed_tuned", trc)
        put("trueStFixM", "stanley_tuned_matched", "pp_fixed_tuned_matched", trc)
        fa = pd.read_csv(FINAL / "front_axle.csv").groupby(["controller", "ref"]).mean(numeric_only=True).groupby("controller").mean()
        for c, key in (("stanley_tuned_matched", "StM"), ("pp_fixed_tuned_matched", "FixM")):
            d[f"front{key}"], d[f"rear{key}"] = 100 * fa.loc[c, "rmse_front"], 100 * fa.loc[c, "rmse_rear"]
    # dynamic single-track check (experiments/dynamic_check.py)
    if (FINAL / "dynamic" / "episodes.csv").exists():
        dy = pd.read_csv(FINAL / "dynamic" / "episodes.csv")
        dy["completed"] = dy["completed"].astype(str).str.lower().eq("true").astype(float)
        dl = dy.groupby(["model", "controller", "train_seed", "ref"]).mean(numeric_only=True).groupby(
            ["model", "controller", "train_seed"]).mean(numeric_only=True)
        for mdl, mk in (("dynamic_norate", "Dt"), ("dynamic", "Dr")):
            for c, key in (("pp_fixed_tuned_matched", "Fix"), ("rlpp_joint_matched", "RL")):
                v = dl.loc[(mdl, c)]
                d[f"rmse{mk}{key}"] = 100 * float(v.rmse.mean())
                d[f"comp{mk}{key}"] = 100 * float(v.completed.mean())
                if len(v) > 1:
                    d[f"rmse{mk}{key}Lo"], d[f"rmse{mk}{key}Hi"] = 100 * float(v.rmse.min()), 100 * float(v.rmse.max())
        dc = pd.read_csv(FINAL / "dynamic" / "comparisons.csv").set_index(["A", "B"])
        put("dynTRLFix", "dynamic_norate:rlpp_joint_matched", "dynamic_norate:pp_fixed_tuned_matched", dc)
        put("dynRRLFix", "dynamic:rlpp_joint_matched", "dynamic:pp_fixed_tuned_matched", dc)

    # dynamic-model retuning (experiments/tune_extra.py, robustness_eval.py, dynamic_stats.py)
    rb = FINAL / "robustness"
    if (rb / "dynamic_summary.csv").exists():
        ds = pd.read_csv(rb / "dynamic_summary.csv").set_index(["experiment", "controller"])
        dc = pd.read_csv(rb / "dynamic_comparisons.csv")
        for exp, mk in (("dynamic_dt", "Dt"), ("dynamic_dr", "Dr")):
            for c, key in (("fixed_retuned", "FixR"), ("fixed_retuned_highg", "FixH"), ("stanley_retuned", "StR"),
                           ("rlpp_retrained", "RLR")):
                r = ds.loc[(exp, c)]
                d[f"rmse{mk}{key}"] = float(r.rmse_cm)
                d[f"trueRmse{mk}{key}"] = float(r.rmse_true_cm)
                d[f"comp{mk}{key}"] = float(r.completed_pct)
            for b, key in (("stanley_retuned", "St"), ("rlpp_retrained", "RL")):
                r = dc[(dc.experiment == exp) & (dc.A == "fixed_retuned") & (dc.B == b) & (dc.metric == "rmse")].iloc[0]
                pre = f"dyn{mk}Fix{key}"
                d[f"{pre}Diff"], d[f"{pre}Lo"], d[f"{pre}Hi"] = float(r["diff"] if "diff" in r else r.diff_cm), float(r.lo), float(r.hi)
                d[f"{pre}Better"], d[f"{pre}P"] = int(r.A_better_on), float(r.p)
        d["bppGainHighDr"] = float(json.loads((RUNS / "extra_tuning" / "fixedhigh_dr.json").read_text())["best"]["params"][1])
        dynamic_table(ds)

    lines = [f"\\newcommand{{\\VfrozenKL}}{{{kl_frozen_text}}}"]   # approx. KL while the actor is frozen
    INT = ("rew", "nEp", "comp", "nRefs", "nInst", "nSynth", "nRerun", "ekSel", "ekFail", "selSame", "selTotal")
    for k, v in d.items():
        if k.endswith("Better") or k.startswith("nPrim"):
            s = f"{v:.0f}"
        elif k in ("stK", "stKs", "stKm", "stKsm"):
            s = f"{v:g}"
        elif k.endswith("P") and k.startswith(("cmp", "true", "dyn")):
            s = f"{v:.4f}"
        elif k in ("klLo", "klHi", "rankCorr") or k.endswith("score") or k.startswith("adRmse"):
            s = f"{v:.3f}"
        elif k.startswith(("diag", "exRew")) and k.endswith(("rew", "zero", "noise")) or k.startswith("exT"):
            s = f"{v:.0f}"
        elif k.startswith(("scale", "latAcc", "within", "gPPO", "tExpert", "tFix", "tBpp", "sp")):
            s = f"{v:.1f}"
        elif k in ("fixL", "fixLm", "bppL", "adA", "adAm", "adB", "adBm"):
            s = f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"
        elif k in ("fixG", "fixGm", "bppG"):
            s = f"{v:.1f}"
        elif k == "bcSel":
            s = f"{v:g}"
        elif k.startswith(INT) or k in ("rlppLband", "latAccFrac"):
            s = f"{v:.0f}"
        else:
            s = (f"{v:.2f}" if v >= 0 else f"$-${-v:.2f}") if isinstance(v, float) else str(v)
        lines.append(f"\\newcommand{{\\V{k}}}{{{s}}}")  # V prefix avoids clashes with class internals
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    circuits_table(); paper_circuits_table(); shapes_table(); comparisons_table()
    true_centerline_table(); spacing_table(); numbers()
    print("tables ->", OUT)
