"""
Build the summary table of all approaches and methods for the paper
(paper/tables/summary_all_methods.tex).

One row per configuration, grouped by method/approach. Every planned
configuration has a row; cells without results yet are shown as "--", so the
table fills itself in when new results are added to results/ and this script
is run again (see method_comparison.py for where each method's files go).

Columns
    Positive videos:  median latency [Q1, Q3] (s) over all runs with an alarm,
                      premature alarms (% of runs with an alarm),
                      videos without any alarm (missed)
    Negative videos:  false alarms (% of runs, i.e. of videos for 1 run/video)
"""

from pathlib import Path

import numpy as np

from method_comparison import load_all

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "paper" / "tables" / "summary_all_methods.tex"

CONFS = [0.125, 0.25, 0.50, 0.75]
RULES = [(1, 1), (3, 2), (5, 3)]
DASH = "--"


def c(x):
    return f"{x:g}"


def cshow(x):
    # as printed in the table: 0.125, 0.25, 0.50, 0.75
    return "0.125" if x == 0.125 else f"{x:.2f}"


# ------------------------------------------------------------------
# Planned rows: (block title, [(row cells for the parameter columns, key)])
# key = (method, approach, config label as produced by method_comparison)
# ------------------------------------------------------------------

def planned_rows():
    sf = "SAFERFIRE"
    blocks = [
        ("Light Only", [
            ((cshow(cl), "", f"({w},{t})"), (sf, "Light Only", f"c={c(cl)} ({w},{t})"))
            for cl in CONFS for (w, t) in RULES]),
        ("Best Only", [
            (("", cshow(ch), ""), (sf, "Best Only", f"c={c(ch)}"))
            for ch in CONFS]),
        ("Two-Step", [
            ((cshow(cl), cshow(ch), f"({w},{t})"), (sf, "Two-Step", f"cL={c(cl)} cH={c(ch)} ({w},{t})"))
            for cl in CONFS for ch in CONFS if ch >= cl for (w, t) in RULES]),
        (r"Ven\^{a}ncio et al.~\cite{venancio2023hybrid} (YOLOv5l + AVT)", [
            ((rf"\multicolumn{{3}}{{l}}{{$c={cf}$, area ${int(a * 100)}\%$, window {win}}}",),
             ("Venancio et al. (2023)", "YOLOv5l (fine-tuned) + AVT", f"c={cf} area={a:g} win={win}"))
            for cf, a, win in [("0.25", 0.05, 20), ("0.25", 0.10, 30), ("0.4", 0.05, 20), ("0.5", 0.05, 30)]]),
        # onfire_baseline/launch_simulation_batches_onfire.py: the two modes of the original work,
        # run separately, alarm on the first frame classified as fire (no temporal rule)
        (r"OnFire~\cite{thomson2020onfire} (ShuffleNetV2-OnFire)", [
            ((r"\multicolumn{3}{l}{full frame, $P(\mathrm{fire})\geq0.5$}",),
             ("OnFire (Thomson et al. 2020)", "Full frame", "thr=0.5 (1,1)")),
            ((r"\multicolumn{3}{l}{superpixels, $\geq1$ with $P(\mathrm{fire})\geq0.5$}",),
             ("OnFire (Thomson et al. 2020)", "Superpixels", "thr=0.5 k=1")),
        ]),
        # launch_simulation_flame.py, edge regime: YOLOv8s detector (Tc), background filter (T_fg)
        # and tracking with confirmation time t_C
        (r"FLAME~\cite{gragnaniello2025flame} (YOLOv8s + motion analysis)", [
            ((rf"\multicolumn{{3}}{{l}}{{$c=0.25$, $T_{{fg}}={fg}$, $t_C={tc}$~s}}",),
             ("FLAME (Gragnaniello et al. 2025)", "Edge", f"c=0.25 fg={float(fg):g} tC={float(tc):g}"))
            for fg, tc in [("0.20", "1.0"), ("0.20", "0.5"), ("0.30", "1.0")]]),
        # experiments/dumitru_*.py: YOLOv10n and an ablation of the 5-level validation cascade
        # (confidence fire >= 0.40 / smoke >= 0.55, size, HSV colour, K consecutive inspected frames, cooldown)
        (r"Dumitru and Sp\^{\i}nu~\cite{dumitru2026multitask} (YOLOv10n + validation cascade)", [
            ((rf"\multicolumn{{3}}{{l}}{{{desc}}}",),
             ("Dumitru & Spinu (2026)", "YOLOv10n + validation cascade", key))
            for desc, key in [("confidence only, $K=1$", "raw K=1"),
                              ("+ size, $K=1$", "size K=1"),
                              ("+ size + color, $K=1$", "size_hsv K=1"),
                              ("+ size, $K=3$", "size K=3"),
                              ("full cascade, $K=3$", "full K=3"),
                              ("full cascade, $K=5$", "full K=5")]]),
    ]
    return blocks


# ------------------------------------------------------------------
# Statistics per configuration
# ------------------------------------------------------------------

def stats(data):
    out = {}
    pos, neg = data[data.kind == "pos"], data[data.kind == "neg"]
    for key, g in pos.groupby(["method", "approach", "config"]):
        v = g["value"].to_numpy(float)
        alarms = v[np.isfinite(v)]
        per_video = g.groupby("video")["value"].apply(lambda s: s.notna().any())
        out[key] = {
            "lat": (np.median(alarms), *np.percentile(alarms, [25, 75])) if len(alarms) else None,
            "prem": 100 * np.mean(alarms < 0) if len(alarms) else None,
            "missed": int((~per_video).sum()),
        }
    for key, g in neg.groupby(["method", "approach", "config"]):
        out.setdefault(key, {})["neg"] = 100 * g["value"].mean()
    return out


def fmt_row(params, s):
    if s is None:
        s = {}
    lat = s.get("lat")
    cells = list(params) + [
        f"{lat[0]:.2f} [{lat[1]:.2f}, {lat[2]:.2f}]" if lat else DASH,
        f"{s['prem']:.2f}" if s.get("prem") is not None else DASH,
        f"{s['missed']}" if "missed" in s else DASH,
        f"{s['neg']:.2f}" if s.get("neg") is not None else DASH,
    ]
    return " & ".join(cells) + r" \\"


HEADER = r"""
\begingroup
\small
\renewcommand{\baselinestretch}{1}\selectfont   % single spacing in the table (review mode doubles it)
\setlength{\tabcolsep}{6pt}
\setlength{\LTcapwidth}{\textwidth}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{ccc@{\hspace{14pt}}rrr@{\hspace{14pt}}r}
\caption{\hl{Summary of all evaluated configurations of the proposed approaches and of the methods from the literature, on the 69 positive and 69 negative videos. Latency: median and interquartile range (Q1, Q3) over all runs with an alarm (negative values: alarm before the annotated onset). Premature: percentage of runs with an alarm in which the alarm preceded the onset. Missed: positive videos without any alarm. False alarms: percentage of negative-video runs with an alarm. Empty cells: not applicable.}}
\label{tab:summary_all_methods}\\
\toprule
 & & & \multicolumn{3}{c}{\textbf{Positive videos}} & \textbf{Negative videos} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-7}
$c_L$ & $c_H$ & $(W,T)$ & \makecell{Latency (s)\\median [Q1, Q3]} & \makecell{Premature\\(\%)} & \makecell{Missed\\(videos)} & \makecell{False alarms\\(\%)} \\
\midrule
\endfirsthead
\multicolumn{7}{l}{\footnotesize\itshape Table~\ref{tab:summary_all_methods} (continued)}\\
\toprule
$c_L$ & $c_H$ & $(W,T)$ & \makecell{Latency (s)\\median [Q1, Q3]} & \makecell{Premature\\(\%)} & \makecell{Missed\\(videos)} & \makecell{False alarms\\(\%)} \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{\footnotesize\itshape continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
"""

FOOTER = r"""\end{longtable}
\endgroup
"""


def main():
    st = stats(load_all())
    lines = [
        "% Generated by analysis/summary_table.py -- do not edit by hand.",
        HEADER.strip("\n"),
    ]
    first = True
    for title, rows in planned_rows():
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{title}}}}} \\*")
        prev = None
        for params, key in rows:
            if key[0] == "SAFERFIRE" and key[1] != "Best Only":
                cl, ch, rule = params
                # space between (c_L, c_H) groups; their values shown only on the group's first row
                if prev is not None and (cl, ch) != prev:
                    lines.append(r"\addlinespace[5pt]")
                params = (cl, ch, rule) if (cl, ch) != prev else ("", "", rule)
                prev = (cl, ch)
            lines.append(fmt_row(params, st.get(key)))
    lines.append(FOOTER)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    filled = sum(1 for _, rows in planned_rows() for _, k in rows if k in st)
    total = sum(len(rows) for _, rows in planned_rows())
    print(f"wrote {OUT}  ({filled}/{total} rows with results)")


if __name__ == "__main__":
    main()
