"""Builds the reference-path cache and the train/val/test split manifest.

1. Rasterises every F1TENTH centerline with the repository's own `create_track_images.py`
   (1200 px, 7 px stroke) and runs the unchanged image->path front end in strict mode.
2. Measures reference-extraction error against the source centerline (known exactly,
   because the rasteriser's pixel mapping is deterministic).
3. Extracts synthetic shapes/curves; failures are recorded, never dropped silently.
4. Assigns circuits to splits with a fixed, documented rule (paper circuits -> test,
   remaining circuits by a seeded permutation), before any model is trained.

Usage:  python experiments/prepare_data.py
"""
from __future__ import annotations
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CACHE, MANIFEST  # noqa: E402
from environment import BeadEnvironment, PathExtractionError  # noqa: E402
from geometry_utils import points_to_polyline_distance  # noqa: E402
from create_track_images import generate_track_image  # noqa: E402

CIRCUIT_DIR = ROOT / "datasets" / "f1tenth_circuits"
SPLIT_SEED = 20261011
PAPER_CIRCUITS = ["Hockenheim", "Montreal", "YasMarina"]
N_EXTRA_TEST, N_VAL = 5, 4
SYNTH = {  # (family, source split, how many, assigned split)
    "closed_train": ("closed_shape", "closed_shapes_dataset/train", 60, "train"),
    "curve_train": ("open_curve", "curve_dataset/train", 60, "train"),
    "closed_val": ("closed_shape", "closed_shapes_dataset/val", 25, "val"),
    "curve_val": ("open_curve", "curve_dataset/val", 25, "val"),
    "closed_test": ("closed_shape", "closed_shapes_dataset/test", 150, "test"),
    "curve_test": ("open_curve", "curve_dataset/test", 150, "test"),
}


def assign_circuits(names: list[str]) -> dict[str, tuple[str, str]]:
    others = sorted(n for n in names if n not in PAPER_CIRCUITS)
    perm = list(np.random.default_rng(SPLIT_SEED).permutation(others))
    out = {n: ("test", "paper circuit; held out from all training and model selection") for n in PAPER_CIRCUITS}
    for n in perm[:N_EXTRA_TEST]:
        out[n] = ("test", f"seeded permutation (seed {SPLIT_SEED}), positions 1-{N_EXTRA_TEST}")
    for n in perm[N_EXTRA_TEST:N_EXTRA_TEST + N_VAL]:
        out[n] = ("val", f"seeded permutation (seed {SPLIT_SEED}), next {N_VAL}")
    for n in perm[N_EXTRA_TEST + N_VAL:]:
        out[n] = ("train", f"seeded permutation (seed {SPLIT_SEED}), remainder")
    return out


def centerline_in_arena(csv: Path) -> np.ndarray:
    """Exact image of the source centerline in arena coordinates (mirrors the rasteriser)."""
    pts = pd.read_csv(csv, comment="#", header=None).iloc[:, :2].values.astype(float)
    margin, size = 80, 1200
    sc = (size - 2 * margin) / max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    px = margin + (pts[:, 0] - pts[:, 0].min()) * sc
    py = (size - margin) - (pts[:, 1] - pts[:, 1].min()) * sc
    s = 2 * BeadEnvironment.BOUNDARY / (size - 1)
    gt = np.stack([(px - (size - 1) / 2) * s, ((size - 1) / 2 - py) * s], 1)
    return np.vstack([gt, gt[:1]])


def extract(args):
    pid, img = args
    try:
        env = BeadEnvironment(image_path=str(img), seed=0, strict=True)
    except PathExtractionError as e:
        return pid, None, None, str(e)
    return pid, env.path_points, sorted(env.path_breaks), None


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    img_dir = CIRCUIT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    csvs = sorted((CIRCUIT_DIR / "centerlines").glob("*_centerline.csv"))
    names = [c.stem.replace("_centerline", "") for c in csvs]
    assign = assign_circuits(names)

    jobs, meta = [], {}
    for csv, name in zip(csvs, names):
        img = img_dir / f"{name}.png"
        generate_track_image(str(csv), str(img))
        pid = f"circuit_{name}"
        jobs.append((pid, img))
        meta[pid] = {"family": "f1tenth_circuit", "source": str(img.relative_to(ROOT)).replace("\\", "/"),
                     "split": assign[name][0], "reason": assign[name][1], "csv": csv}
    for key, (fam, rel, n, split) in SYNTH.items():
        files = sorted((ROOT / "datasets" / rel).glob("*.jpg"))[:n]
        for f in files:
            pid = f"{fam}_{f.stem}"
            jobs.append((pid, f))
            meta[pid] = {"family": fam, "source": str(f.relative_to(ROOT)).replace("\\", "/"), "split": split,
                         "reason": f"first {n} files (sorted) of the generator's {rel.split('/')[-1]} split"}

    with Pool(16) as pool:
        results = pool.map(extract, jobs)

    entries = []
    for pid, pts, brk, err in results:
        m = meta[pid]
        e = {"id": pid, "family": m["family"], "source": m["source"], "split": m["split"],
             "reason": m["reason"], "extraction_ok": err is None, "error": err}
        if err is None:
            np.savez_compressed(CACHE / f"{pid}.npz", points=pts, breaks=np.asarray(brk, int))
            L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            e.update({"n_points": int(len(pts)), "n_breaks": len(brk), "length_m": round(L, 3),
                      "closure_gap_m": round(float(np.linalg.norm(pts[0] - pts[-1])), 4)})
            if "csv" in m:
                gt = centerline_in_arena(m["csv"])
                d_ext = points_to_polyline_distance(pts, gt)
                d_gt = points_to_polyline_distance(gt, pts, path_breaks=set(brk))
                e["extraction_error_m"] = {"ref_to_gt_mean": round(float(d_ext.mean()), 4),
                                           "ref_to_gt_max": round(float(d_ext.max()), 4),
                                           "gt_to_ref_mean": round(float(d_gt.mean()), 4),
                                           "gt_to_ref_max": round(float(d_gt.max()), 4)}
        entries.append(e)

    manifest = {
        "description": "Reference-path split. Circuits are split by whole circuit; synthetic images are "
                       "i.i.d. draws from the same generator in every split (not a geometry-family hold-out).",
        "split_seed": SPLIT_SEED,
        "circuit_source": "github.com/f1tenth/f1tenth_racetracks @ "
                          + (CIRCUIT_DIR / "SOURCE_COMMIT.txt").read_text().split()[0],
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    for split in ("train", "val", "test"):
        es = [e for e in entries if e["split"] == split]
        fams = {}
        for e in es:
            fams.setdefault(e["family"], [0, 0])
            fams[e["family"]][0 if e["extraction_ok"] else 1] += 1
        print(split, {k: f"{v[0]} ok / {v[1]} failed" for k, v in fams.items()})
    print("circuits:", {s: [n for n in names if assign[n][0] == s] for s in ("train", "val", "test")})
    print("manifest ->", MANIFEST)


if __name__ == "__main__":
    main()
