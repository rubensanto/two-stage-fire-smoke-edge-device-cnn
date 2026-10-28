"""
Ideal-model latency vs. Eq. (4) vs. observed latency (Section 4.5.3).

The ideal latencies come from experiments/ideal_latency.py (results/positive_videos/ideal_latency.csv): the simulated
stream of Algorithm 1 replayed with ideal models (a frame is detected if and only if it is at or after the annotated
onset) and constant inference times t_L = 0.184 s and t_H = 1.371 s, one deterministic value per video.
The observed latencies are those of results/positive_videos/saferfire_latency.xlsx; Two-Step is compared with
c_L = c_H = c, Best Only with c_H = c and Light Only with c_L = c.

Output:
    paper/tables/ideal_latency.tex            Table 9 (written directly, like the other table scripts)
    analysis/output/ideal_vs_observed_latency.pdf (and .png)   Figure 16c (copied to paper/figs by make_all.py)
The values quoted in Section 4.5.3 are printed.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configurations import (CONFIDENCES, OUTPUT, PAPER, POSITIVE_RESULTS, RESULTS, RULES, fmt_conf,  # noqa: E402
                            font_family, sheet_name)

IDEAL_CSV = RESULTS / "positive_videos" / "ideal_latency.csv"
OUT_TABLE = PAPER / "tables" / "ideal_latency.tex"
OUT_FIG = OUTPUT / "ideal_vs_observed_latency.pdf"

T_L, T_H = 0.184, 1.371

# approaches of the table: (key, label, approach, rule)
ROWS = ([("LO", r, "Light Only", r) for r in RULES] + [("BO", None, "Best Only", None)]
        + [("TS", r, "Two-Step", r) for r in RULES])

TEXT_WIDTH_IN = 6.53
PAPER_RC = {
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.color": "#D9D9D9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
}
CONF_COLORS = {0.125: "#E69F00", 0.25: "#CC79A7", 0.50: "#0072B2", 0.75: "#009E73"}   # as c_H in Figure 16b
IDEAL_COLOR = "#7F7F7F"
EQ_COLOR = "black"
PREMATURE_COLOR = "#D55E00"
YMIN, YMAX = -25, 60   # limits of the figure; the number of per-video medians outside them is written beyond the axis


def rule_text(rule):
    return f"({rule[0]},{rule[1]})"


def name(approach, rule):
    return "Best Only" if approach == "Best Only" else f"{approach} {rule_text(rule)}"


def expected(key, rule):
    """Eq. (4) of the paper (Light Only: same waiting time and T lightweight inferences, no verification)."""
    if key == "BO":
        return 1.5 * T_H
    T = rule[1]
    return (T + 0.5) * T_L + (T_H if key == "TS" else 0.0)


def observed_sheet(approach, rule, c):
    if approach == "Light Only":
        return sheet_name(approach, rule, c_L=c)
    if approach == "Best Only":
        return sheet_name(approach, c_H=c)
    return sheet_name(approach, rule, c_L=c, c_H=c)


def main():
    ideal = pd.read_csv(IDEAL_CSV)
    ideal.index = ideal["video"].str.rsplit(".", n=1).str[0]

    observed = {}   # (approach name, c) -> DataFrame runs x videos
    for _, _, approach, rule in ROWS:
        for c in CONFIDENCES:
            df = pd.read_excel(POSITIVE_RESULTS, observed_sheet(approach, rule, c)).drop(columns="run")
            missing = set(df.columns) ^ set(ideal.index)
            assert not missing, f"videos differ between the workbook and {IDEAL_CSV.name}: {missing}"
            observed[(name(approach, rule), c)] = df[ideal.index]

    # ---------------- values quoted in the text ----------------
    print(f"{len(ideal)} videos, FPS: {sorted(ideal['fps'].round(3).unique())}")
    bo = ideal["Best Only"]
    for key, rule, approach, _ in ROWS:
        v = ideal[name(approach, rule)]
        print(f"ideal {name(approach, rule):<18} mean {v.mean():.2f} median {v.median():.2f} "
              f"[{v.min():.2f}, {v.max():.2f}]  Eq.(4) {expected(key, rule):.2f}  "
              f"mean - Eq. {v.mean() - expected(key, rule):+.3f}")
        if key == "TS":
            faster, equal = int((v < bo).sum()), int((v == bo).sum())
            print(f"      faster than Best Only in {faster} ({100 * faster / len(v):.2f}%) videos, equal in {equal}, "
                  f"median BO - TS {(bo - v).median():+.2f} s")
    for c in CONFIDENCES:
        for key, rule, approach, _ in ROWS:
            n = name(approach, rule)
            runs = observed[(n, c)]
            pooled = runs.to_numpy(float).ravel()
            pooled = pooled[np.isfinite(pooled)]
            delay = (runs.median() - ideal[n]).dropna()   # per-video median minus ideal latency
            print(f"c={fmt_conf(c)} {n:<18} observed median {np.median(pooled):6.2f}  ideal median "
                  f"{ideal[n].median():.2f}  per-video (observed median - ideal): median {delay.median():6.2f} "
                  f"IQR [{delay.quantile(.25):.2f}, {delay.quantile(.75):.2f}]  share of observed median "
                  f"{100 * ideal[n].median() / np.median(pooled):.0f}%")

    write_table(ideal, observed)
    draw_figure(ideal, observed)


def write_table(ideal, observed):
    lines = [
        "% Generated by analysis/ideal_latency.py -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption[Ideal-model and observed latencies.]{\hl{Latency of the three approaches with ideal models "
        r"(detection in every frame at or after the annotated onset, inference times $t_L=0.184$~s and "
        r"$t_H=1.371$~s), compared with Eq.}~\eqref{eq:expected_latency}\hl{ and with the observed latencies on the "
        r"69 positive videos. Ideal: median and range over the videos (one deterministic value per video). "
        r"Observed: median over all runs with an alarm, as in Table}~\ref{tab:summary_all_methods}\hl{, for the "
        r"confidence threshold $c$ of the column (Two-Step with $c_L=c_H=c$). For Light Only, the value of "
        r"Eq.}~\eqref{eq:expected_latency}\hl{ is $(T+\tfrac{1}{2})\,t_L$.}}",
        r"\label{tab:ideal_latency}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r" & & & & \multicolumn{4}{c}{Observed median (s), $c$} \\",
        r"\cmidrule(lr){5-8}",
        r"Approach & $(W,T)$ & \makecell{Eq.~\eqref{eq:expected_latency}\\(s)} & \makecell{Ideal (s)\\median [min, max]} & "
        r"0.125 & 0.25 & 0.50 & 0.75 \\",
        r"\midrule",
    ]
    previous = None
    for key, rule, approach, _ in ROWS:
        n = name(approach, rule)
        v = ideal[n]
        cells = []
        for c in CONFIDENCES:
            pooled = observed[(n, c)].to_numpy(float).ravel()
            pooled = pooled[np.isfinite(pooled)]
            cells.append(f"{np.median(pooled):.2f}")
        if previous is not None and approach != previous:
            lines.append(r"\addlinespace[3pt]")
        label = approach if approach != previous else ""
        lines.append(f"{label} & {rule_text(rule) if rule else '--'} & {expected(key, rule):.2f} & "
                     f"{v.median():.2f} [{v.min():.2f}, {v.max():.2f}] & " + " & ".join(cells) + r" \\")
        previous = approach
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    OUT_TABLE.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {OUT_TABLE}")


def draw_figure(ideal, observed):
    groups = [name("Best Only", None)] + [name("Two-Step", r) for r in RULES]
    keys = [("BO", None)] + [("TS", r) for r in RULES]
    series = [("ideal", None)] + [("obs", c) for c in CONFIDENCES]
    width = 0.15
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * (width + 0.02)

    mpl.rcParams["font.family"] = font_family()
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))
        for g, (group, (key, rule)) in enumerate(zip(groups, keys)):
            for off, (kind, c) in zip(offsets, series):
                values = (ideal[group] if kind == "ideal" else observed[(group, c)].median()).dropna().to_numpy()
                color = IDEAL_COLOR if kind == "ideal" else CONF_COLORS[c]
                x = g + off
                box = ax.boxplot(values, positions=[x], widths=width, patch_artist=True,
                                 showfliers=True, whis=1.5,
                                 medianprops=dict(color="black", linewidth=1.0),
                                 whiskerprops=dict(color="0.25", linewidth=0.6),
                                 capprops=dict(color="0.25", linewidth=0.6),
                                 flierprops=dict(marker="o", markersize=2.0, markerfacecolor="none",
                                                 markeredgecolor="0.45", markeredgewidth=0.4),
                                 zorder=3)
                box["boxes"][0].set_facecolor(mpl.colors.to_rgba(color, 0.55))
                box["boxes"][0].set_edgecolor(color)
                box["boxes"][0].set_linewidth(0.7)
                if (values > YMAX).any():   # above the axis
                    ax.annotate(f"↑{int((values > YMAX).sum())}", (x, YMAX), xytext=(0, 1.5),
                                textcoords="offset points", ha="center", va="bottom", fontsize=6, color="0.3",
                                annotation_clip=False)
                if (values < YMIN).any():   # beside the box, at the lower limit
                    ax.annotate(f"↓{int((values < YMIN).sum())}", (x + width / 2, YMIN), xytext=(1.5, 1.5),
                                textcoords="offset points", ha="left", va="bottom", fontsize=6, color="0.3")
            ax.scatter(g + offsets[0], expected(key, rule), marker="D", s=14, facecolor="white",
                       edgecolor=EQ_COLOR, linewidth=0.7, zorder=5)

        ymin = YMIN
        ax.axhspan(ymin, 0, facecolor=PREMATURE_COLOR, alpha=0.10, edgecolor="none", zorder=0)
        ax.axhline(0, color=PREMATURE_COLOR, linewidth=0.7, zorder=2)
        ax.text(0.995, 0.015, "Premature alarm", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                style="italic", color=PREMATURE_COLOR, zorder=4)
        ax.set_ylim(ymin, YMAX)
        ax.set_xlim(-0.55, len(groups) - 0.45)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylabel("Latency per video (s)")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        handles = ([Patch(facecolor=mpl.colors.to_rgba(IDEAL_COLOR, 0.55), edgecolor=IDEAL_COLOR, label="Ideal models")]
                   + [Patch(facecolor=mpl.colors.to_rgba(CONF_COLORS[c], 0.55), edgecolor=CONF_COLORS[c],
                            label=f"Observed, $c$ = {fmt_conf(c)}") for c in CONFIDENCES]
                   + [Line2D([], [], ls="", marker="D", markersize=4, markerfacecolor="white",
                             markeredgecolor=EQ_COLOR, label="Eq. (4)")])
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=6, frameon=False,
                  handlelength=1.2, columnspacing=1.0)
        OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(OUT_FIG.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.02)
        print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
