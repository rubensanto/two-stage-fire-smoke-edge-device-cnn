import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager


# ============================================================
# CONFIGURATION
# ============================================================

# Input workbook containing the aggregated results.
INPUT_FILE = Path("results_aggregated.xlsx")

# Original comparison table is based on the "Mean (s)" sheet.
INPUT_SHEET = "Mean (s)"

# Output Excel file with the comparison results.
OUTPUT_FILE = Path("comparison_results.xlsx")

# Folder where all figures will be saved.
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

# The original analysis compares 69 videos.
# Video ID 70 ("run") is explicitly excluded.
TOTAL_VIDEOS = 69


# ============================================================
# LATIN MODERN FONT
# ============================================================

# found in the MiKTeX / TeX Live installation (see configs.latin_modern_font)
from configurations import font_family

mpl.rcParams.update({
    "font.family": font_family(),
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 21,
})


# ============================================================
# PAPER STYLE FOR THE DIRECT SPEED COMPARISONS (Figures 4 and 5)
#
# Drawn at the printed size (full \textwidth = 472 pt), so the font
# sizes are the sizes in the PDF. One panel per confidence threshold
# of the lightweight model c_L; inside each panel every point is
# encoded without text labels (no overlaps):
#   marker      = (W,T) rule of the lightweight stage
#   colour      = confidence threshold of the high-capacity model c_H
# ============================================================

from matplotlib.lines import Line2D

from configurations import CONFIGS as ALL_CONFIGS, CONFIDENCES, fmt_conf
from configurations import ids as config_ids
from configurations import label as config_label

TEXT_WIDTH_IN = 6.53

PAPER_RC = {
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
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

RULE_MARKERS = {1: "o", 3: "s", 5: "^"}          # keyed by W
RULE_LABELS = {1: "(1,1)", 3: "(3,2)", 5: "(5,3)"}
HEAVY_COLORS = {                                  # keyed by c_H (Okabe-Ito)
    0.125: "#E69F00",
    0.25: "#CC79A7",
    0.50: "#0072B2",
    0.75: "#009E73",
}


def direct_speed_figure(
    points,
    y_label,
    upper_label,
    filename,
    ideal=None,
):
    """
    points: list of dicts with keys
        x, y, W, c_light, c_heavy
    One panel per c_light (in increasing order).
    ideal: optional {W: (x, y)} of the ideal models, which do not depend
        on the thresholds and are drawn as open black markers in every panel.
    """

    lights = sorted({p["c_light"] for p in points})

    with mpl.rc_context(PAPER_RC):

        fig, axes = plt.subplots(
            1, len(lights),
            figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN / len(lights) + 0.75),
            sharey=True,
            squeeze=False
        )
        axes = axes[0]

        for ax, c_light in zip(axes, lights):

            # Region where the compared approach is faster more often
            ax.fill_between([0, 100], [0, 100], [100, 100], color="0.93", zorder=0, linewidth=0)
            ax.plot([0, 100], [0, 100], linestyle="--", color="0.45", linewidth=0.7, zorder=1)
            if c_light == lights[0]:
                ax.text(3, 97, upper_label, ha="left", va="top", fontsize=6.5,
                        style="italic", color="0.3")

            for p in points:
                if p["c_light"] != c_light:
                    continue
                color = HEAVY_COLORS[p["c_heavy"]]
                ax.scatter(
                    p["x"],
                    p["y"],
                    marker=RULE_MARKERS[p["W"]],
                    s=26,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.95,
                    zorder=3
                )

            for w, (x, y) in (ideal or {}).items():
                ax.scatter(x, y, marker=RULE_MARKERS[w], s=26, facecolors="none",
                           edgecolors="black", linewidths=0.8, zorder=4)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")
            ax.set_xticks(range(0, 101, 25))
            ax.set_yticks(range(0, 101, 25))
            ax.grid(zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(rf"$c_L$ = {fmt_conf(c_light)}", fontsize=8, pad=3)
            ax.set_xlabel("Best Only faster (%)")

        axes[0].set_ylabel(y_label)

        # ----------------------------------------------------
        # Legend below the panels
        # ----------------------------------------------------

        rules = sorted({p["W"] for p in points})
        heavies = sorted({p["c_heavy"] for p in points})
        handles = [
            Line2D([], [], linestyle="none", marker=RULE_MARKERS[w], markersize=4.5,
                   markerfacecolor="0.35", markeredgecolor="0.35", label=RULE_LABELS[w])
            for w in rules
        ] + [
            Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                   markerfacecolor=HEAVY_COLORS[c], markeredgecolor=HEAVY_COLORS[c],
                   label=rf"$c_H$ = {fmt_conf(c)}")
            for c in heavies
        ]
        if ideal:
            handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                                  markerfacecolor="none", markeredgecolor="black",
                                  markeredgewidth=0.8, label="Ideal models"))
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(handles),
            frameon=False,
            handletextpad=0.3,
            columnspacing=1.2,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)

        stem = FIGURE_DIR / filename
        for ext in ("pdf", "png"):
            fig.savefig(
                f"{stem}.{ext}",
                bbox_inches="tight",
                pad_inches=0.02
            )
        plt.close(fig)

    print(f"Saved {stem}.pdf and {stem}.png")


# ============================================================
# FIGURE DISPLAY LABELS AND CONFIGURATIONS
#
# Defined once in configs.py (all confidence thresholds). Internal
# IDs are the column names of the "Mean (s)" sheet written by
# table_results.py.
# ============================================================

FIGURE_LABELS = {cid: config_label(cid) for cid in ALL_CONFIGS}


def ideal_two_step_points():
    """
    {W: (Best Only faster %, Two-Step faster %)} with ideal models
    (results/positive_videos/ideal_latency.csv, experiments/ideal_latency.py):
    same criterion as the observed points, independent of the thresholds.
    """
    from configurations import RESULTS

    ideal = pd.read_csv(RESULTS / "positive_videos" / "ideal_latency.csv")
    best = ideal["Best Only"]
    points = {}
    for w, label in RULE_LABELS.items():
        two = ideal[f"Two-Step {label}"]
        both_valid = (best > 0) & (two > 0)
        points[w] = (pct((both_valid & (best < two)).sum()), pct((both_valid & (two < best)).sum()))
        print(f"Ideal models {label}: Best Only faster {points[w][0]:.2f}%, Two-Step faster {points[w][1]:.2f}%")
    return points


def _description(cid):
    c = ALL_CONFIGS[cid]
    d = {"type": c["approach"], "config": f'{c["group"]}:{c["sheet"]}'}
    if c["approach"] == "Two-Step":
        d.update(low_conf=c["c_L"], N=c["rule"][0], M=c["rule"][1], heavy_conf=c["c_H"])
    elif c["approach"] == "Light Only":
        d.update(light_conf=c["c_L"], N=c["rule"][0], M=c["rule"][1])
    else:
        d.update(heavy_conf=c["c_H"])
    return d


CONFIGURATIONS = {
    cid: _description(cid)
    for cid in config_ids("Best Only") + config_ids("Two-Step")
}

LIGHT_CONFIGURATIONS = {
    cid: _description(cid)
    for cid in config_ids("Light Only")
}

# Best Only vs Two-Step: every Two-Step configuration is compared with
# the Best Only configuration that uses the same c_H.
COMPARISONS = {
    best: config_ids("Two-Step", c_H=ALL_CONFIGS[best]["c_H"])
    for best in config_ids("Best Only")
}


# ============================================================
# READ EXCEL FILE
# ============================================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)


# ============================================================
# VALIDATE VIDEO ID COLUMN
# ============================================================

if "Video ID" not in df.columns:
    raise ValueError(
        "The column 'Video ID' does not exist."
    )


# ============================================================
# KEEP THE ORIGINAL 69-VIDEO POPULATION
# ============================================================

df = df[
    df["Video ID"] <= TOTAL_VIDEOS
].copy()

if len(df) != TOTAL_VIDEOS:
    raise ValueError(
        f"Expected {TOTAL_VIDEOS} videos after filtering, "
        f"but found {len(df)}."
    )


# ============================================================
# VALIDATE BEST ONLY / TWO-STEP COLUMNS
# ============================================================

for best_only, two_step_list in COMPARISONS.items():

    if best_only not in df.columns:
        raise ValueError(
            f"Missing column: {best_only}"
        )

    for two_step in two_step_list:

        if two_step not in df.columns:
            raise ValueError(
                f"Missing column: {two_step}"
            )


# ============================================================
# VALIDATE LIGHT ONLY COLUMNS
# ============================================================

for light_only in LIGHT_CONFIGURATIONS:

    if light_only not in df.columns:
        raise ValueError(
            f"Missing Light Only column: {light_only}"
        )


# ============================================================
# HELPER
# ============================================================

def pct(value):
    """
    Convert a count to a percentage of the
    69-video dataset.
    """
    return round(
        value / TOTAL_VIDEOS * 100,
        2
    )


# ============================================================
# CALCULATE BEST ONLY vs TWO-STEP RESULTS
# ============================================================

results = []


for best_only, two_step_list in COMPARISONS.items():

    for two_step in two_step_list:

        best = df[best_only]
        two = df[two_step]

        # ----------------------------------------------------
        # CLASSIFY EACH RESULT
        #
        # > 0  = valid alarm / valid latency
        # < 0  = false alarm
        # NaN  = no alarm
        # ----------------------------------------------------

        best_valid = best > 0
        best_false_alarm = best < 0
        best_no_alarm = best.isna()

        two_valid = two > 0
        two_false_alarm = two < 0
        two_no_alarm = two.isna()

        # ----------------------------------------------------
        # VALID SPEED COMPARISON
        # ----------------------------------------------------

        both_valid = (
            best_valid &
            two_valid
        )

        two_faster = (
            both_valid &
            (two < best)
        ).sum()

        best_faster = (
            both_valid &
            (best < two)
        ).sum()

        equal = (
            both_valid &
            (best == two)
        ).sum()

        both_valid_count = both_valid.sum()

        speed_advantage_pp = (
            pct(two_faster) -
            pct(best_faster)
        )

        # ----------------------------------------------------
        # ALARM / NO-ALARM DIFFERENCES
        # ----------------------------------------------------

        best_alarm_two_no_alarm = (
            best_valid &
            two_no_alarm
        ).sum()

        two_alarm_best_no_alarm = (
            two_valid &
            best_no_alarm
        ).sum()

        # ----------------------------------------------------
        # FALSE ALARM DIFFERENCES
        # ----------------------------------------------------

        best_false_two_not_false = (
            best_false_alarm &
            ~two_false_alarm
        ).sum()

        two_false_best_not_false = (
            two_false_alarm &
            ~best_false_alarm
        ).sum()

        both_false_alarm = (
            best_false_alarm &
            two_false_alarm
        ).sum()

        best_false_count = best_false_alarm.sum()
        two_false_count = two_false_alarm.sum()

        best_no_alarm_count = best_no_alarm.sum()
        two_no_alarm_count = two_no_alarm.sum()

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            # Identifiers
            "Best Only": best_only,

            "Best Only Configuration":
                CONFIGURATIONS[best_only]["config"],

            "Two-Step": two_step,

            "Two-Step Configuration":
                CONFIGURATIONS[two_step]["config"],

            # Two-Step parameters
            "Low Model Confidence":
                CONFIGURATIONS[two_step]["low_conf"],

            "Window N":
                CONFIGURATIONS[two_step]["N"],

            "Required M":
                CONFIGURATIONS[two_step]["M"],

            "Heavy Model Confidence":
                CONFIGURATIONS[two_step]["heavy_conf"],

            # Dataset
            "Total Videos":
                TOTAL_VIDEOS,

            # Valid alarms / latency
            "Both Valid Alarms":
                both_valid_count,

            "Both Valid Alarms (%)":
                pct(both_valid_count),

            "Two-Step Faster":
                two_faster,

            "Two-Step Faster (%)":
                pct(two_faster),

            "Best Only Faster":
                best_faster,

            "Best Only Faster (%)":
                pct(best_faster),

            "Two-Step Speed Advantage (percentage points)":
                round(speed_advantage_pp, 2),

            # Alarm differences
            "Best Only Alarmed, Two-Step Did Not":
                best_alarm_two_no_alarm,

            "Best Only Alarmed, Two-Step Did Not (%)":
                pct(best_alarm_two_no_alarm),

            "Two-Step Alarmed, Best Only Did Not":
                two_alarm_best_no_alarm,

            "Two-Step Alarmed, Best Only Did Not (%)":
                pct(two_alarm_best_no_alarm),

            # False alarms
            "Best Only False Alarms":
                best_false_count,

            "Best Only False Alarms (%)":
                pct(best_false_count),

            "Two-Step False Alarms":
                two_false_count,

            "Two-Step False Alarms (%)":
                pct(two_false_count),

            "Both False Alarms":
                both_false_alarm,

            "Both False Alarms (%)":
                pct(both_false_alarm),

            "Best Only False Alarmed, Two-Step Did Not":
                best_false_two_not_false,

            "Best Only False Alarmed, Two-Step Did Not (%)":
                pct(best_false_two_not_false),

            "Two-Step False Alarmed, Best Only Did Not":
                two_false_best_not_false,

            "Two-Step False Alarmed, Best Only Did Not (%)":
                pct(two_false_best_not_false),

            # No alarm
            "Best Only Did Not Alarm":
                best_no_alarm_count,

            "Best Only Did Not Alarm (%)":
                pct(best_no_alarm_count),

            "Two-Step Did Not Alarm":
                two_no_alarm_count,

            "Two-Step Did Not Alarm (%)":
                pct(two_no_alarm_count),

            # Equal valid latencies
            "Equal Valid Latencies":
                equal,
        })


results_df = pd.DataFrame(results)


# ============================================================
# PREPARE BEST ONLY vs TWO-STEP FIGURE DATA
# ============================================================

results_df["N/M"] = (
    results_df["Window N"].astype(str)
    + "/"
    + results_df["Required M"].astype(str)
)


results_df["Confidence Group"] = (
    "Best conf. "
    + results_df["Best Only"].map(
        lambda b: fmt_conf(CONFIGURATIONS[b]["heavy_conf"])
    )
    + " | Two-Step low conf. "
    + results_df["Low Model Confidence"].map(
        lambda x: f"{x:.2f}"
    )
)


results_df["Config Label"] = (
    results_df["Best Only"].map(FIGURE_LABELS)
    + " → "
    + results_df["Two-Step"].map(FIGURE_LABELS)
)


# ============================================================
# FIGURE 1 — SPEED ADVANTAGE HEATMAP
# ============================================================

heatmap_1 = results_df[
    results_df["Best Only"] == "B1"
].copy()


heatmap_2 = results_df[
    results_df["Best Only"] == "B2"
].copy()


# ------------------------------------------------------------
# B1
# ------------------------------------------------------------

pivot_b1 = heatmap_1.pivot(
    index="Low Model Confidence",
    columns="N/M",
    values="Two-Step Speed Advantage (percentage points)"
)


# ------------------------------------------------------------
# B2
# ------------------------------------------------------------

pivot_b2 = heatmap_2[
    heatmap_2["Low Model Confidence"] == 0.25
].pivot(
    index="Low Model Confidence",
    columns="N/M",
    values="Two-Step Speed Advantage (percentage points)"
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 6.5),
    constrained_layout=True
)


# Common colour range.

values = pd.concat([
    pivot_b1.stack(),
    pivot_b2.stack()
])


vmax = max(
    abs(values.min()),
    abs(values.max())
)


for ax, pivot, title in [

    (
        axes[0],
        pivot_b1,
        "Best Only confidence = 0.25 | "
        "Two-Step high confidence = 0.25"
    ),

    (
        axes[1],
        pivot_b2,
        "Best Only confidence = 0.50 | "
        "Two-Step low confidence = 0.25, "
        "high confidence = 0.50"
    ),

]:

    image = ax.imshow(
        pivot.values,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        pivot.columns
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        [
            f"{x:.2f}"
            for x in pivot.index
        ]
    )

    ax.set_xlabel(
        "Two-Step N/M"
    )

    ax.set_ylabel(
        "Two-Step low confidence"
    )

    ax.set_title(
        title
    )

    for row in range(
        pivot.shape[0]
    ):

        for col in range(
            pivot.shape[1]
        ):

            value = pivot.iloc[
                row,
                col
            ]

            ax.text(
                col,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center"
            )


cbar = fig.colorbar(
    image,
    ax=axes,
    shrink=0.95
)


cbar.set_label(
    "Two-Step faster − Best Only faster "
    "(percentage points)"
)


fig.suptitle(
    "Speed advantage of Two-Step relative to Best Only",
    fontsize=14
)


fig.savefig(
    FIGURE_DIR /
    "01_speed_advantage_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 2 — SPEED ADVANTAGE VS VALIDITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


for _, row in results_df.iterrows():

    marker = (
        "o"
        if row["Best Only"] == "B1"
        else "s"
    )

    ax.scatter(
        row[
            "Two-Step Speed Advantage "
            "(percentage points)"
        ],
        row["Both Valid Alarms (%)"],
        marker=marker,
        s=90
    )

    ax.annotate(
        row["Config Label"],
        (
            row[
                "Two-Step Speed Advantage "
                "(percentage points)"
            ],
            row["Both Valid Alarms (%)"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Two-Step speed advantage "
    "(percentage points)"
)


ax.set_ylabel(
    "Both valid (%)"
)


ax.set_title(
    "Speed advantage versus validity"
)


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR /
    "02_speed_vs_validity.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 3 — ALARM / FALSE-ALARM COMPARISON
# ============================================================

alarm_plot = results_df[
    [
        "Config Label",
        "Best Only False Alarms (%)",
        "Two-Step False Alarms (%)",
        "Best Only Alarmed, Two-Step Did Not (%)",
        "Two-Step Alarmed, Best Only Did Not (%)"
    ]
].copy()


fig, ax = plt.subplots(
    figsize=(12, 6)
)


x = range(
    len(alarm_plot)
)


width = 0.20


ax.bar(
    [
        i - 1.5 * width
        for i in x
    ],
    alarm_plot[
        "Best Only False Alarms (%)"
    ],
    width=width,
    label="Best Only false alarms"
)


ax.bar(
    [
        i - 0.5 * width
        for i in x
    ],
    alarm_plot[
        "Two-Step False Alarms (%)"
    ],
    width=width,
    label="Two-Step false alarms"
)


ax.bar(
    [
        i + 0.5 * width
        for i in x
    ],
    alarm_plot[
        "Best Only Alarmed, Two-Step Did Not (%)"
    ],
    width=width,
    label="Best Only alarmed / Two-Step did not"
)


ax.bar(
    [
        i + 1.5 * width
        for i in x
    ],
    alarm_plot[
        "Two-Step Alarmed, Best Only Did Not (%)"
    ],
    width=width,
    label="Two-Step alarmed / Best Only did not"
)


ax.set_xticks(
    list(x)
)


ax.set_xticklabels(
    alarm_plot["Config Label"],
    rotation=45,
    ha="right"
)


ax.set_ylabel(
    "Percentage of all videos (%)"
)


ax.set_title(
    "Alarm and false-alarm behaviour by configuration"
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR /
    "03_alarm_behaviour.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# FIGURE 4 — DIRECT SPEED COMPARISON
#              BEST ONLY vs TWO-STEP
#
# Each Two-Step configuration is compared with the Best Only
# configuration that uses the same high-capacity threshold c_H.
# ============================================================

direct_speed_figure(
    points=[
        {
            "x": row["Best Only Faster (%)"],
            "y": row["Two-Step Faster (%)"],
            "W": CONFIGURATIONS[row["Two-Step"]]["N"],
            "c_light": CONFIGURATIONS[row["Two-Step"]]["low_conf"],
            "c_heavy": CONFIGURATIONS[row["Two-Step"]]["heavy_conf"],
        }
        for _, row in results_df.iterrows()
    ],
    y_label="Two-Step faster (% of videos)",
    upper_label="Two-Step faster\nmore often",
    filename="pairwise_best_only_vs_two_step",
    ideal=ideal_two_step_points(),
)


# ============================================================
# BEST ONLY vs LIGHT ONLY CALCULATIONS
# ============================================================

best_vs_light_results = []


for best_only in config_ids("Best Only"):

    best = df[best_only]

    best_valid = best > 0
    best_false_alarm = best < 0
    best_no_alarm = best.isna()


    for light_only, light_config in LIGHT_CONFIGURATIONS.items():

        light = df[light_only]

        light_valid = light > 0
        light_false_alarm = light < 0
        light_no_alarm = light.isna()


        # ----------------------------------------------------
        # VALID SPEED COMPARISON
        # ----------------------------------------------------

        both_valid = (
            best_valid &
            light_valid
        )


        light_faster = (
            both_valid &
            (light < best)
        ).sum()


        best_faster = (
            both_valid &
            (best < light)
        ).sum()


        equal = (
            both_valid &
            (best == light)
        ).sum()


        both_valid_count = both_valid.sum()


        # Positive:
        # Light Only faster more often.
        #
        # Negative:
        # Best Only faster more often.

        speed_advantage_pp = (
            pct(light_faster) -
            pct(best_faster)
        )


        # ----------------------------------------------------
        # ALARM / NO-ALARM DIFFERENCES
        # ----------------------------------------------------

        best_alarm_light_no_alarm = (
            best_valid &
            light_no_alarm
        ).sum()


        light_alarm_best_no_alarm = (
            light_valid &
            best_no_alarm
        ).sum()


        # ----------------------------------------------------
        # FALSE ALARM DIFFERENCES
        # ----------------------------------------------------

        best_false_light_not_false = (
            best_false_alarm &
            ~light_false_alarm
        ).sum()


        light_false_best_not_false = (
            light_false_alarm &
            ~best_false_alarm
        ).sum()


        both_false_alarm = (
            best_false_alarm &
            light_false_alarm
        ).sum()


        best_false_count = (
            best_false_alarm.sum()
        )


        light_false_count = (
            light_false_alarm.sum()
        )


        best_no_alarm_count = (
            best_no_alarm.sum()
        )


        light_no_alarm_count = (
            light_no_alarm.sum()
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        best_vs_light_results.append({

            # Identifiers
            "Best Only":
                best_only,

            "Best Only Configuration":
                CONFIGURATIONS[
                    best_only
                ]["config"],

            "Light Only":
                light_only,

            "Light Only Configuration":
                light_config["config"],

            # Light Only parameters
            "Light Model Confidence":
                light_config["light_conf"],

            "Window N":
                light_config["N"],

            "Required M":
                light_config["M"],

            # Dataset
            "Total Videos":
                TOTAL_VIDEOS,

            # Valid speed comparison
            "Both Valid":
                both_valid_count,

            "Both Valid (%)":
                pct(
                    both_valid_count
                ),

            "Light Only Faster":
                light_faster,

            "Light Only Faster (%)":
                pct(
                    light_faster
                ),

            "Best Only Faster":
                best_faster,

            "Best Only Faster (%)":
                pct(
                    best_faster
                ),

            "Light Only Speed Advantage "
            "(percentage points)":
                round(
                    speed_advantage_pp,
                    2
                ),

            "Equal Valid Latencies":
                equal,

            # Alarm differences
            "Best Only Alarmed, Light Only Did Not":
                best_alarm_light_no_alarm,

            "Best Only Alarmed, Light Only Did Not (%)":
                pct(
                    best_alarm_light_no_alarm
                ),

            "Light Only Alarmed, Best Only Did Not":
                light_alarm_best_no_alarm,

            "Light Only Alarmed, Best Only Did Not (%)":
                pct(
                    light_alarm_best_no_alarm
                ),

            # False alarms
            "Best Only False Alarms":
                best_false_count,

            "Best Only False Alarms (%)":
                pct(
                    best_false_count
                ),

            "Light Only False Alarms":
                light_false_count,

            "Light Only False Alarms (%)":
                pct(
                    light_false_count
                ),

            "Both False Alarms":
                both_false_alarm,

            "Both False Alarms (%)":
                pct(
                    both_false_alarm
                ),

            "Best Only False Alarmed, Light Only Did Not":
                best_false_light_not_false,

            "Best Only False Alarmed, Light Only Did Not (%)":
                pct(
                    best_false_light_not_false
                ),

            "Light Only False Alarmed, Best Only Did Not":
                light_false_best_not_false,

            "Light Only False Alarmed, Best Only Did Not (%)":
                pct(
                    light_false_best_not_false
                ),

            # No alarm
            "Best Only Did Not Alarm":
                best_no_alarm_count,

            "Best Only Did Not Alarm (%)":
                pct(
                    best_no_alarm_count
                ),

            "Light Only Did Not Alarm":
                light_no_alarm_count,

            "Light Only Did Not Alarm (%)":
                pct(
                    light_no_alarm_count
                ),
        })


best_vs_light_df = pd.DataFrame(
    best_vs_light_results
)


# ============================================================
# LIGHT ONLY FIGURE LABEL
# ============================================================

best_vs_light_df["Config Label"] = (
    best_vs_light_df["Best Only"].map(
        FIGURE_LABELS
    )
    + " → "
    + best_vs_light_df["Light Only"].map(
        FIGURE_LABELS
    )
)


# ============================================================
# FIGURE 5 — DIRECT SPEED COMPARISON
#              BEST ONLY vs LIGHT ONLY
#
# Every Light Only configuration is compared with every Best Only
# configuration (panel = c_L of Light Only, colour = c_H of Best Only).
# ============================================================

direct_speed_figure(
    points=[
        {
            "x": row["Best Only Faster (%)"],
            "y": row["Light Only Faster (%)"],
            "W": row["Window N"],
            "c_light": row["Light Model Confidence"],
            "c_heavy": CONFIGURATIONS[row["Best Only"]]["heavy_conf"],
        }
        for _, row in best_vs_light_df.iterrows()
    ],
    y_label="Light Only faster (% of videos)",
    upper_label="Light Only faster\nmore often",
    filename="pairwise_light_only_vs_best_only",
)


# ============================================================
# SAVE EXCEL RESULTS
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # Best Only vs Two-Step
    # --------------------------------------------------------

    results_df.to_excel(
        writer,
        sheet_name="Comparison",
        index=False
    )


    # --------------------------------------------------------
    # Figure data for Best Only vs Two-Step
    # --------------------------------------------------------

    results_df[
        [
            "Config Label",
            "Best Only",
            "Two-Step",
            "Low Model Confidence",
            "Window N",
            "Required M",
            "Heavy Model Confidence",
            "Both Valid Alarms (%)",
            "Two-Step Faster (%)",
            "Best Only Faster (%)",
            "Two-Step Speed Advantage "
            "(percentage points)",
            "Best Only False Alarms (%)",
            "Two-Step False Alarms (%)",
            "Best Only Alarmed, Two-Step Did Not (%)",
            "Two-Step Alarmed, Best Only Did Not (%)"
        ]
    ].to_excel(
        writer,
        sheet_name="Figure Data",
        index=False
    )


    # --------------------------------------------------------
    # Best Only vs Light Only
    # --------------------------------------------------------

    best_vs_light_df.to_excel(
        writer,
        sheet_name="Best vs Light",
        index=False
    )


    # --------------------------------------------------------
    # Figure data for Best Only vs Light Only
    # --------------------------------------------------------

    best_vs_light_df[
        [
            "Config Label",
            "Best Only",
            "Light Only",
            "Light Model Confidence",
            "Window N",
            "Required M",
            "Both Valid (%)",
            "Light Only Faster (%)",
            "Best Only Faster (%)",
            "Light Only Speed Advantage "
            "(percentage points)",
            "Best Only False Alarms (%)",
            "Light Only False Alarms (%)",
            "Best Only Alarmed, Light Only Did Not (%)",
            "Light Only Alarmed, Best Only Did Not (%)"
        ]
    ].to_excel(
        writer,
        sheet_name="Light Figure Data",
        index=False
    )


# ============================================================
# COMPACT TERMINAL OUTPUT
# ============================================================

print()

print("=" * 72)
print("BEST ONLY vs TWO-STEP")
print("=" * 72)


for _, r in results_df.iterrows():

    print()

    print(
        f"{r['Config Label']}"
    )

    print("-" * 72)

    print(
        f"Both valid:             "
        f"{r['Both Valid Alarms']:>2} "
        f"({r['Both Valid Alarms (%)']:>5.2f}%)"
    )

    print(
        f"Two-Step faster:        "
        f"{r['Two-Step Faster']:>2} "
        f"({r['Two-Step Faster (%)']:>5.2f}%)"
    )

    print(
        f"Best Only faster:       "
        f"{r['Best Only Faster']:>2} "
        f"({r['Best Only Faster (%)']:>5.2f}%)"
    )

    print(
        f"Speed advantage TS:     "
        f"{r['Two-Step Speed Advantage (percentage points)']:+6.2f} pp"
    )

    print(
        f"Best false alarms:      "
        f"{r['Best Only False Alarms']:>2} "
        f"({r['Best Only False Alarms (%)']:>5.2f}%)"
    )

    print(
        f"Two-Step false alarms:  "
        f"{r['Two-Step False Alarms']:>2} "
        f"({r['Two-Step False Alarms (%)']:>5.2f}%)"
    )


# ============================================================
# BEST ONLY vs LIGHT ONLY TERMINAL OUTPUT
# ============================================================

print()

print("=" * 72)
print("BEST ONLY vs LIGHT ONLY")
print("=" * 72)


for _, r in best_vs_light_df.iterrows():

    print()

    print(
        f"{r['Config Label']}"
    )

    print("-" * 72)

    print(
        f"Both valid:             "
        f"{r['Both Valid']:>2} "
        f"({r['Both Valid (%)']:>5.2f}%)"
    )

    print(
        f"Best Only faster:       "
        f"{r['Best Only Faster']:>2} "
        f"({r['Best Only Faster (%)']:>5.2f}%)"
    )

    print(
        f"Light Only faster:      "
        f"{r['Light Only Faster']:>2} "
        f"({r['Light Only Faster (%)']:>5.2f}%)"
    )

    print(
        f"Speed advantage Light:  "
        f"{r['Light Only Speed Advantage (percentage points)']:+6.2f} pp"
    )

    print(
        f"Best false alarms:      "
        f"{r['Best Only False Alarms']:>2} "
        f"({r['Best Only False Alarms (%)']:>5.2f}%)"
    )

    print(
        f"Light false alarms:     "
        f"{r['Light Only False Alarms']:>2} "
        f"({r['Light Only False Alarms (%)']:>5.2f}%)"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 72)

print(
    f"Saved Excel: {OUTPUT_FILE}"
)

print(
    f"Saved figures: {FIGURE_DIR.resolve()}"
)

print("=" * 72)