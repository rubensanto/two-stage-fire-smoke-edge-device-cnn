"""
Configurations of the system-level evaluation (Light Only, Best Only and
Two-Step) and the paths of the repository: single source of truth for every
script in analysis/.

Every configuration has a short ID (L1-L12, B1-B4, T1-T30, used in the
internal tables), its parameters and the sheet where its runs are stored in
results/positive_videos/saferfire_latency.xlsx (and, for the false alarms,
in results/negative_videos/saferfire_false_alarms.xlsx). The sheet names are
the same in both workbooks and are also the names written by
experiments/saferfire_positive.py and experiments/saferfire_negative.py:

    LO_c0.25_W1T1            Light Only,  c = 0.25, (W,T) = (1,1)
    BO_c0.50                 Best Only,   c = 0.50
    TS_cL0.125_cH0.75_W5T3   Two-Step,    c_L = 0.125, c_H = 0.75, (W,T) = (5,3)

Each sheet has one row per run ("run" column) and one column per video
(latency in s; empty = no alarm; < 0 = premature alarm).
To add runs, use analysis/import_runs.py.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAPER = REPO / "paper"
RESULTS = REPO / "results"
DATA = REPO / "data"
MODELS = REPO / "models"
OUTPUT = HERE / "output"                      # generated files (not versioned)
# videos are not versioned: data/videos/{positive,negative}/ or SAFERFIRE_VIDEOS
VIDEOS = Path(os.environ.get("SAFERFIRE_VIDEOS", DATA / "videos"))

POSITIVE_RESULTS = RESULTS / "positive_videos" / "saferfire_latency.xlsx"
NEGATIVE_RESULTS = RESULTS / "negative_videos" / "saferfire_false_alarms.xlsx"

RESULT_FILES = {"saferfire": [POSITIVE_RESULTS]}

CONFIDENCES = [0.125, 0.25, 0.50, 0.75]
RULES = [(1, 1), (3, 2), (5, 3)]


def fmt_conf(c):
    return "0.125" if c == 0.125 else f"{c:.2f}"


def sheet_name(approach, rule=None, c_L=None, c_H=None):
    """Sheet name of a configuration (same convention as the experiment scripts)."""
    if approach == "Light Only":
        return f"LO_c{fmt_conf(c_L)}_W{rule[0]}T{rule[1]}"
    if approach == "Best Only":
        return f"BO_c{fmt_conf(c_H)}"
    return f"TS_cL{fmt_conf(c_L)}_cH{fmt_conf(c_H)}_W{rule[0]}T{rule[1]}"


def _build():
    configs = {}

    def add(cid, approach, rule, c_light, c_heavy):
        configs[cid] = {
            "approach": approach,
            "rule": rule,                 # (W, T) or None for Best Only
            "c_L": c_light,
            "c_H": c_heavy,
            "group": "saferfire",
            "sheet": sheet_name(approach, rule, c_light, c_heavy),
        }

    # ---- confidence thresholds 0.25 / 0.50 (IDs used since the first revision) ----
    k = 1
    for c in (0.25, 0.50):
        for rule in RULES:
            add(f"L{k}", "Light Only", rule, c, None)
            k += 1
    add("B1", "Best Only", None, None, 0.25)
    add("B2", "Best Only", None, None, 0.50)
    main_ts = [  # (ID, c_L, rule, c_H)
        ("T1", 0.25, (1, 1), 0.25), ("T2", 0.25, (1, 1), 0.50),
        ("T3", 0.25, (3, 2), 0.25), ("T4", 0.25, (3, 2), 0.50),
        ("T5", 0.25, (5, 3), 0.25), ("T6", 0.25, (5, 3), 0.50),
        ("T7", 0.50, (1, 1), 0.50),
        ("T8", 0.50, (3, 2), 0.50),
        ("T9", 0.50, (5, 3), 0.50),
    ]
    for cid, cl, rule, ch in main_ts:
        add(cid, "Two-Step", rule, cl, ch)

    # ---- additional confidence thresholds 0.125 / 0.75 ----
    for c in (0.125, 0.75):
        for rule in RULES:
            add(f"L{k}", "Light Only", rule, c, None)
            k += 1
    add("B3", "Best Only", None, None, 0.125)
    add("B4", "Best Only", None, None, 0.75)
    k = 10
    for cl, chs in [(0.125, [0.125, 0.25, 0.50, 0.75]), (0.25, [0.75]), (0.50, [0.75]), (0.75, [0.75])]:
        for ch in chs:
            for rule in RULES:
                add(f"T{k}", "Two-Step", rule, cl, ch)
                k += 1
    return configs


CONFIGS = _build()

SHEET_TO_ID = {(c["group"], c["sheet"]): cid for cid, c in CONFIGS.items()}



def rule_label(rule):
    return f"({rule[0]},{rule[1]})"


def label(cid):
    """Compact label, e.g. L[1,1](0.25), B(0.5), T[3,2](0.125, 0.75)."""
    c = CONFIGS[cid]
    if c["approach"] == "Best Only":
        return f"B({c['c_H']:g})"
    w, t = c["rule"]
    if c["approach"] == "Light Only":
        return f"L[{w},{t}]({c['c_L']:g})"
    return f"T[{w},{t}]({c['c_L']:g}, {c['c_H']:g})"


def ids(approach=None, c_L=None, c_H=None):
    """IDs of the matching configurations, ordered by c_L, c_H and (W,T)."""
    out = [
        cid for cid, c in CONFIGS.items()
        if (approach is None or c["approach"] == approach)
        and (c_L is None or c["c_L"] == c_L)
        and (c_H is None or c["c_H"] == c_H)
    ]
    return sorted(out, key=lambda i: (
        CONFIGS[i]["c_L"] or 0, CONFIGS[i]["c_H"] or 0, CONFIGS[i]["rule"] or (0, 0)))


def best_only_for(c_H):
    return ids("Best Only", c_H=c_H)[0]


def latin_modern_font():
    """
    Path of the Latin Modern Roman font used in the paper figures
    (lmroman10-regular.otf from MiKTeX or TeX Live), or None if it is not
    found; set the environment variable LATIN_MODERN_FONT to override.
    """
    import os
    import shutil
    import subprocess

    name = "lmroman10-regular.otf"
    candidates = [
        os.environ.get("LATIN_MODERN_FONT", ""),
        Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm" / name,
        Path("C:/Program Files/MiKTeX/fonts/opentype/public/lm") / name,
        Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm") / name,
        Path("/usr/share/texmf/fonts/opentype/public/lm") / name,
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    if shutil.which("kpsewhich"):
        found = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
        if found:
            return found
    return None


def font_family():
    """Register Latin Modern with matplotlib and return its family name ("serif" if missing)."""
    from matplotlib import font_manager

    path = latin_modern_font()
    if path is None:
        print("WARNING: Latin Modern font not found; using the default serif font")
        return "serif"
    font_manager.fontManager.addfont(path)
    return font_manager.FontProperties(fname=path).get_name()
