"""Builds one self-contained folder (and zip) for Overleaf or journal upload: paper/ijcas_submission.

The manuscripts normally read tables and figures from ../revised/. Here every file they use is copied
next to them and the paths are rewritten, so a recompile elsewhere cannot pick up old figure files.
FIGURE_CHECK.txt lists which file each figure of the full manuscript uses.

Usage: python paper/scripts/build_submission.py
"""
from __future__ import annotations
import re
import shutil
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parents[1]
SRC, REV = P / "ijcas", P / "revised"
OUT = P / "ijcas_submission"
TEX = ["main_full.tex", "main.tex", "supplementary.tex"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tables").mkdir(parents=True)
    (OUT / "figures").mkdir()
    for f in ("IJCAS.cls", "picins.sty", "published-look.sty"):
        shutil.copy2(SRC / f, OUT / f)
    used_fig, used_tab = set(), set()
    for name in TEX:
        s = (SRC / name).read_text(encoding="utf-8")
        used_tab |= set(re.findall(r"\.\./revised/tables/([\w.]+)", s))
        used_fig |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([\w.]+)\}", s))
        s = s.replace("../revised/tables/", "tables/").replace("\\graphicspath{{../revised/figures/}}", "\\graphicspath{{figures/}}")
        assert "../revised" not in s, name
        (OUT / name).write_text(s, encoding="utf-8")
    for t in used_tab:
        shutil.copy2(REV / "tables" / t, OUT / "tables" / t)
    for g in used_fig:
        shutil.copy2(REV / "figures" / g, OUT / "figures" / g)
    full = (OUT / "main_full.tex").read_text(encoding="utf-8")
    pairs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([\w.]+)\}", full)
    (OUT / "FIGURE_CHECK.txt").write_text(
        "Figures of main_full.tex, in order:\n" + "\n".join(f"Fig. {i + 1}: figures/{n}" for i, n in enumerate(pairs)) + "\n",
        encoding="utf-8")
    z = P / "ijcas_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(OUT))
    print("->", OUT, "|", z, "|", len(used_tab), "tables,", len(used_fig), "figures")


if __name__ == "__main__":
    main()
