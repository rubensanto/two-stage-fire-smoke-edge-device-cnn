"""
Ideal-case analytical prediction vs. observed results for Best Only vs. Two-Step,
per video (mean latency over the runs of each video, as in pairwise_comparison.py / Figure 16b).

Model (Section 3.5, p_L = p_H = 1, random event onset):
    L_BO = U_BO + t_H,          U_BO ~ U[0, t_H]
    L_2S = U_2S + T t_L + t_H,  U_2S ~ U[0, t_L]
    E[L_BO - L_2S]      = t_H/2 - (T + 1/2) t_L
    sd of the difference of the means of n runs = sqrt((t_H^2 + t_L^2) / (12 n))
    P(mean L_2S < mean L_BO) ~= Phi(E / sd)                          (Eq. 6)
A Monte Carlo evaluation of the same model is printed as a check.

Observed values use the same data and criterion as pairwise_comparison.py / Figure 16b:
mean latency over the runs of each video, only videos where both means are
> 0, percentage of the 69 videos. The configurations (all Two-Step
configurations, with every c_L and c_H) are read from configurations.py; the
prediction is drawn for the number of runs of each configuration (14 for
c = 0.25/0.50, 6 for 0.125/0.75).

Output: analysis/output/theory_vs_observed.pdf (and .png); the values are quoted in Section 4.5.3.
"""

import sys
from math import erf, sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configurations import CONFIGS, OUTPUT, RESULT_FILES, best_only_for, fmt_conf, font_family, ids  # noqa: E402

# values quoted in Section 4.5.3; the figure itself is no longer in the paper
OUT_PDF = OUTPUT / "theory_vs_observed.pdf"

T_L, T_H = 0.184, 1.371
N_VIDEOS = 69
N_MC = 400_000

mpl.rcParams["font.family"] = font_family()

TEXT_WIDTH_IN = 6.53
PAPER_RC = {
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.linewidth": 0.4,
    "grid.color": "#D9D9D9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
}
RULE_LABELS = {1: "(1,1)", 2: "(3,2)", 3: "(5,3)"}           # keyed by T
LIGHT_MARKERS = {0.125: "D", 0.25: "o", 0.50: "s", 0.75: "^"}  # keyed by c_L
HEAVY_COLORS = {0.125: "#E69F00", 0.25: "#CC79A7", 0.50: "#0072B2", 0.75: "#009E73"}  # keyed by c_H
PRED_COLOR = "#4D4D4D"
PRED_STYLES = {14: "-", 6: "--"}                               # keyed by runs per video
DIFF_YMIN = -14   # lower limit of panel (b); longer interquartile bars are cut and marked with an arrow


def load(cid):
    c = CONFIGS[cid]
    frames = [pd.read_excel(f, c["sheet"]) for f in RESULT_FILES[c["group"]]]
    return pd.concat(frames).drop(columns="run")


def expected_advantage(T):
    return T_H / 2 - (T + 0.5) * T_L


def predicted_per_video(T, runs):
    sd = sqrt((T_H ** 2 + T_L ** 2) / (12 * runs))
    z = expected_advantage(T) / sd
    return 0.5 * (1 + erf(z / sqrt(2)))


def monte_carlo_per_video(T, runs, rng):
    u_bo = rng.uniform(0, T_H, (N_MC, runs)).mean(axis=1)
    u_2s = (rng.uniform(0, T_L, (N_MC, runs)) + T * T_L).mean(axis=1)
    return float((u_2s < u_bo).mean())


def observed(two_step, best_only):
    # rounded so that exactly equal means (e.g. poss(6)) count as ties, not float noise
    m2, mb = two_step.mean().round(9), best_only.mean().round(9)
    both_valid = (m2 > 0) & (mb > 0)
    return {
        "per_video": float((both_valid & (m2 < mb)).sum() / N_VIDEOS),
        "diff": (mb - m2)[both_valid].to_numpy(),
    }


def main():
    rng = np.random.default_rng(0)
    best = {CONFIGS[b]["c_H"]: load(b) for b in ids("Best Only")}

    rows = []
    for cid in ids("Two-Step"):
        c = CONFIGS[cid]
        two = load(cid)
        runs = int(min(len(two), len(best[c["c_H"]])))
        rows.append({"id": cid, "c_L": c["c_L"], "T": c["rule"][1], "c_H": c["c_H"], "runs": runs,
                     **observed(two, best[c["c_H"]])})

    run_counts = sorted({r["runs"] for r in rows}, reverse=True)
    pred = {(T, n): {"per_video": predicted_per_video(T, n), "diff": expected_advantage(T)}
            for T in RULE_LABELS for n in run_counts}

    for n in run_counts:
        print(f"runs per video: {n}")
        for T in RULE_LABELS:
            p = pred[(T, n)]
            print(f"  pred {RULE_LABELS[T]}: video {100*p['per_video']:.2f}% "
                  f"(Monte Carlo {100*monte_carlo_per_video(T, n, rng):.1f}%)  dE {p['diff']:+.2f} s")
    for r in rows:
        q1, med, q3 = np.percentile(r["diff"], [25, 50, 75])
        print(f"obs  {RULE_LABELS[r['T']]} cL={r['c_L']} cH={r['c_H']} (n={r['runs']}): "
              f"video {100*r['per_video']:.2f}%  diff median {med:+.2f} s (IQR {q1:+.2f}..{q3:+.2f})")

    # configurations of each rule, ordered by c_L and c_H, spread around the rule position
    per_rule = max(sum(1 for r in rows if r["T"] == T) for T in RULE_LABELS)
    offsets = np.linspace(-0.4, 0.4, per_rule)

    with mpl.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))

        panels = [
            ("per_video", "Two-Step faster (% of videos)", "(a) Videos in which Two-Step is faster"),
            ("diff", r"$\bar{L}_{\mathrm{BO}}-\bar{L}_{\mathrm{2S}}$ per video (s)", "(b) Latency difference"),
        ]

        for ax, (key, ylabel, title) in zip(axes, panels):
            ax.grid(axis="y", zorder=0)
            ax.set_title(title, fontsize=8.5, pad=4)
            ax.set_ylabel(ylabel)
            ax.set_xticks(list(RULE_LABELS))
            ax.set_xticklabels(RULE_LABELS.values())
            ax.set_xlim(0.5, 3.5)
            ax.set_xlabel("Temporal rule $(W,T)$")

            for T in RULE_LABELS:
                for n in run_counts:
                    value = pred[(T, n)][key] * (1 if key == "diff" else 100)
                    ax.hlines(value, T - 0.45, T + 0.45, color=PRED_COLOR, lw=1.2,
                              ls=PRED_STYLES.get(n, ":"), zorder=2)
                    if key == "diff":
                        break   # the expected difference does not depend on n

            for T in RULE_LABELS:
                group = sorted((r for r in rows if r["T"] == T), key=lambda r: (r["c_L"], r["c_H"]))
                for off, r in zip(offsets, group):
                    x = T + off
                    color = HEAVY_COLORS[r["c_H"]]
                    if key == "diff":
                        q1, med, q3 = np.percentile(r["diff"], [25, 50, 75])
                        ax.vlines(x, max(q1, DIFF_YMIN), q3, color=color, lw=0.8, zorder=3)
                        if q1 < DIFF_YMIN:   # bar cut at the axis limit
                            ax.scatter(x, DIFF_YMIN + 0.35, s=10, marker="v", color=color, zorder=4,
                                       clip_on=False)
                        y = med
                    else:
                        y = 100 * r[key]
                        ax.vlines(x, y, pred[(T, r["runs"])][key] * 100, color=color, lw=0.5, ls=":",
                                  zorder=2)
                    ax.scatter(x, y, s=18, marker=LIGHT_MARKERS[r["c_L"]], facecolor=color,
                               edgecolor="white", linewidth=0.4, zorder=4)

            if key == "diff":
                ax.axhline(0, color="black", lw=0.6, zorder=1)
                ax.set_ylim(DIFF_YMIN, 3)
                ax.fill_between([0.5, 3.5], 0, 3, color="#EEEEEE", zorder=0)
                ax.text(3.45, 1.5, "Two-Step faster", ha="right", va="center", fontsize=7, color="#555555")
            else:
                ax.set_ylim(0, 105)
                ax.axhline(50, color="#999999", lw=0.5, ls="--", zorder=1)

        handles = [
            Line2D([], [], color=PRED_COLOR, lw=1.2, ls=PRED_STYLES.get(n, ":"),
                   label=f"Ideal-case prediction, n = {n}")
            for n in run_counts
        ] + [
            Line2D([], [], ls="", marker="o", markersize=4.5, color=HEAVY_COLORS[c],
                   label=rf"$c_H$ = {fmt_conf(c)}")
            for c in sorted({r["c_H"] for r in rows})
        ] + [
            Line2D([], [], ls="", marker=LIGHT_MARKERS[c], markersize=4.5, color="0.35",
                   label=rf"$c_L$ = {fmt_conf(c)}")
            for c in sorted({r["c_L"] for r in rows})
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5)
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PDF, bbox_inches="tight")
        fig.savefig(OUT_PDF.with_suffix(".png"), bbox_inches="tight")
        print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
