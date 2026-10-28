"""
Compare SAFERFIRE (Light Only / Best Only / Two-Step) with external methods
(Venancio et al. 2023, OnFire, FLAME, ...) under the same protocol.

Positive videos (latency, s; NaN = no alarm; < 0 = premature alarm) and, when
available, negative videos (1 = false alarm, 0 = no alarm) are loaded into one
long table, summarized per (method, approach, configuration) and plotted.

To add a method, add an entry to METHODS with its result workbooks
(results/positive_videos/<method>_latency.xlsx and
results/negative_videos/<method>_false_alarms.xlsx) and a label function for
its sheet names.

Notes
- The "run" column is an index, not a video, and is never treated as one.

Outputs (analysis/output/):
    method_comparison.xlsx   summary + long table
    method_comparison.png/.pdf
"""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POS, NEG = HERE.parent / "results" / "positive_videos", HERE.parent / "results" / "negative_videos"
OUTPUT = HERE / "output"
N_POS_VIDEOS = 69

# Reference for the pairwise comparison (method, approach, config)
REFERENCE = ("SAFERFIRE", "Best Only", "c=0.25")


# ------------------------------------------------------------------
# Sheet-name parsers -> (approach, config label)
# (sheet names as in results/README.md; the labels are the keys used by
# summary_table.py and false_alarm_tables.py)
# ------------------------------------------------------------------

def label_saferfire(sheet):
    # LO_c0.25_W1T1, BO_c0.50, TS_cL0.125_cH0.75_W5T3
    if m := re.fullmatch(r"LO_c([\d.]+)_W(\d)T(\d)", sheet):
        return "Light Only", f"c={float(m[1]):g} ({m[2]},{m[3]})"
    if m := re.fullmatch(r"BO_c([\d.]+)", sheet):
        return "Best Only", f"c={float(m[1]):g}"
    if m := re.fullmatch(r"TS_cL([\d.]+)_cH([\d.]+)_W(\d)T(\d)", sheet):
        return "Two-Step", f"cL={float(m[1]):g} cH={float(m[2]):g} ({m[3]},{m[4]})"
    return None


def label_venancio(sheet):
    # AVT_c0.25_area0.05_win20: YOLOv5l + area variation technique, window of 20 frames
    if m := re.fullmatch(r"AVT_c([\d.]+)_area([\d.]+)_win(\d+)", sheet):
        return "YOLOv5l (fine-tuned) + AVT", f"c={float(m[1]):g} area={float(m[2]):g} win={int(m[3])}"
    return None


def label_onfire(sheet):
    # the two modes of the OnFire paper, each frame classified on its own (no temporal rule):
    #   FF_thr0.50      full-frame classifier, alarm when P(fire) >= thr
    #   SP_thr0.50_k1   superpixel localisation (SLIC), alarm when at least k superpixels have P(fire) >= thr
    if m := re.fullmatch(r"FF_thr([\d.]+)", sheet):
        return "Full frame", f"thr={float(m[1]):g} (1,1)"
    if m := re.fullmatch(r"SP_thr([\d.]+)_k(\d+)", sheet):
        return "Superpixels", f"thr={float(m[1]):g} k={m[2]}"
    return None


def label_flame(sheet):
    # edge_c0.25_fg0.20_tC1.0: edge regime (virtual clock on the RPi5, as the other methods), confidence Tc,
    # foreground overlap T_fg, confirmation time tC (tO = 0.1 s). The offline_* sheets (every frame,
    # 15 videos) only validate the implementation and are not read here.
    if m := re.fullmatch(r"edge_c([\d.]+)_fg([\d.]+)_tC([\d.]+)", sheet):
        return "Edge", f"c={float(m[1]):g} fg={float(m[2]):g} tC={float(m[3]):g}"
    return None


def label_dumitru(sheet):
    # raw_K1, size_K1, size_hsv_K1, size_K3, full_K3, full_K5: YOLOv10n + the levels of the validation cascade
    # that are active (confidence, size, HSV colour, temporal confirmation over K inspected frames, cooldown)
    if m := re.fullmatch(r"(raw|size|size_hsv|full)_K(\d)", sheet):
        return "YOLOv10n + validation cascade", f"{m[1]} K={m[2]}"
    return None


METHODS = [
    {
        "method": "SAFERFIRE",
        # 14 runs per video with c = 0.25/0.50, 6 with 0.125/0.75; negative videos: 1 run per video
        "pos": [POS / "saferfire_latency.xlsx"],
        "neg": [NEG / "saferfire_false_alarms.xlsx"],
        "label": label_saferfire,
    },
    {
        "method": "Venancio et al. (2023)",
        "pos": [POS / "venancio_latency.xlsx"],
        "neg": [NEG / "venancio_false_alarms.xlsx"],
        "label": label_venancio,
    },
    {
        "method": "OnFire (Thomson et al. 2020)",
        "pos": [POS / "onfire_latency.xlsx"],
        "neg": [NEG / "onfire_false_alarms.xlsx"],
        "label": label_onfire,
    },
    {
        "method": "FLAME (Gragnaniello et al. 2025)",
        "pos": [POS / "flame_latency.xlsx"],
        "neg": [NEG / "flame_false_alarms.xlsx"],
        "label": label_flame,
    },
    {
        "method": "Dumitru & Spinu (2026)",
        "pos": [POS / "dumitru_latency.xlsx"],
        "neg": [NEG / "dumitru_false_alarms.xlsx"],
        "label": label_dumitru,
    },
]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load(files, label, kind):
    rows, run_offset = [], 0
    for f in files:
        if not f.exists():
            continue
        xls = pd.ExcelFile(f)
        n_runs = 0
        for sheet in xls.sheet_names:
            lab = label(sheet)
            if lab is None:
                continue
            df = pd.read_excel(xls, sheet)
            runs = df.pop("run") if "run" in df else pd.Series(range(1, len(df) + 1))
            n_runs = max(n_runs, len(df))
            long = df.assign(run=runs.to_numpy() + run_offset).melt(
                id_vars="run", var_name="video", value_name="value")
            long["approach"], long["config"], long["kind"] = lab[0], lab[1], kind
            rows.append(long)
        run_offset += n_runs
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_all():
    parts = []
    for m in METHODS:
        for kind in ("pos", "neg"):
            d = load(m[kind], m["label"], kind)
            if not d.empty:
                parts.append(d.assign(method=m["method"]))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def wilcoxon_signed_rank(diff):
    """Two-sided Wilcoxon signed-rank test (normal approximation with tie and
    zero corrections). Returns (n, W+, p)."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return n, np.nan, np.nan
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = (w_plus - mean - 0.5 * np.sign(w_plus - mean)) / np.sqrt(var)
    p = 1 + math.erf(-abs(z) / math.sqrt(2))      # = 2 * Phi(-|z|)
    return n, w_plus, p


def per_video_means(pos):
    # rounded so that exactly equal means count as ties, not float noise
    return pos.groupby(["method", "approach", "config", "video"])["value"].mean().round(9)


def summarize(data):
    pos, neg = data[data.kind == "pos"], data[data.kind == "neg"]
    means = per_video_means(pos)
    ref = means.loc[REFERENCE]

    rows = []
    for key, g in pos.groupby(["method", "approach", "config"]):
        v = g["value"].to_numpy(float)
        m = means.loc[key]
        valid = m > 0
        row = {
            "method": key[0], "approach": key[1], "config": key[2],
            "runs_per_video": g.groupby("video")["run"].nunique().max(),
            "videos": g["video"].nunique(),
            # pooled over all runs with an alarm (incl. premature), as in the paper, without the run column
            "median_latency_s": np.nanmedian(v) if np.isfinite(v).any() else np.nan,
            "premature_alarm_rate_pct": 100 * np.sum(v < 0) / np.isfinite(v).sum(),   # of runs with an alarm, as in the paper
            "missed_runs_pct": 100 * np.isnan(v).sum() / len(v),
            "videos_without_alarm": int(m.isna().sum()),
        }
        # pairwise comparison with the reference (per-video mean latency, as compare.py)
        both = valid & (ref.reindex(m.index) > 0)
        diff = (ref.reindex(m.index) - m)[both]          # > 0: method faster
        n, _, p = wilcoxon_signed_rank(diff)
        row.update({
            "faster_than_ref_pct": 100 * (diff > 0).sum() / N_POS_VIDEOS,
            "slower_than_ref_pct": 100 * (diff < 0).sum() / N_POS_VIDEOS,
            "median_diff_vs_ref_s": diff.median() if n else np.nan,
            "wilcoxon_p_vs_ref": p,
        })
        nk = neg[(neg.method == key[0]) & (neg.approach == key[1]) & (neg.config == key[2])]
        row["neg_false_alarm_rate_pct"] = 100 * nk["value"].mean() if len(nk) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Figure: latency vs. false alarms
# ------------------------------------------------------------------

STYLE = {
    ("SAFERFIRE", "Light Only"): ("#E69F00", "o"),
    ("SAFERFIRE", "Best Only"): ("#009E73", "s"),
    ("SAFERFIRE", "Two-Step"): ("#0072B2", "^"),
}
OTHER_COLORS = ["#CC79A7", "#D55E00", "#56B4E9", "#000000"]


def figure(summary, path):
    use_neg = summary["neg_false_alarm_rate_pct"].notna().all()
    ycol = "neg_false_alarm_rate_pct" if use_neg else "premature_alarm_rate_pct"
    ylabel = ("False alarms on negative videos (%)" if use_neg
              else "Premature alarms on positive videos (% of runs)")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    others = [k for k in summary.groupby(["method", "approach"]).groups if k not in STYLE]
    style = {**STYLE, **{k: (OTHER_COLORS[i % 4], "D") for i, k in enumerate(others)}}
    for (method, approach), g in summary.groupby(["method", "approach"]):
        color, marker = style[(method, approach)]
        label = approach if method == "SAFERFIRE" else f"{method}: {approach}"
        ax.scatter(g["median_latency_s"], g[ycol], color=color, marker=marker, s=36,
                   label=label, edgecolor="white", linewidth=0.5, zorder=3)

    # Pareto front (lower latency and lower false-alarm rate are better)
    pts = summary[["median_latency_s", ycol]].dropna().sort_values(["median_latency_s", ycol])
    front, best = [], np.inf
    for x, y in pts.to_numpy():
        if y < best:
            front.append((x, y)); best = y
    if front:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color="#999999", lw=0.8, ls="--", zorder=1, label="Pareto front")

    ax.set_xlabel("Median latency on positive videos (s)")
    ax.set_ylabel(ylabel)
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    data = load_all()
    summary = summarize(data).sort_values(["method", "approach", "median_latency_s"])

    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / "method_comparison.xlsx"
    with pd.ExcelWriter(out) as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        data.to_excel(w, sheet_name="long_table", index=False)
    figure(summary, OUTPUT / "method_comparison")

    pd.set_option("display.width", 200)
    cols = ["method", "approach", "config", "runs_per_video", "median_latency_s",
            "premature_alarm_rate_pct", "videos_without_alarm", "faster_than_ref_pct",
            "slower_than_ref_pct", "median_diff_vs_ref_s", "wilcoxon_p_vs_ref", "neg_false_alarm_rate_pct"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nReference: {REFERENCE}.  Saved {out.name} and method_comparison.png/.pdf")


if __name__ == "__main__":
    main()
