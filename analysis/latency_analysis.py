import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from openpyxl.styles import Font, Alignment
from matplotlib.patches import Patch

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "."

OUTPUT_EXCEL = "results_aggregated.xlsx"

# Number of figures for the video x approach boxplot grids
N_FIGURES = 4

# Original baseline approach
BASELINE_APPROACH = "L1"   # Light Only, c_L = 0.25, (1,1)


# ============================================================
# LATIN MODERN FONT
# ============================================================

# found in the MiKTeX / TeX Live installation (see configs.latin_modern_font)
from configurations import font_family

mpl.rcParams.update({
    "font.family": font_family(),
    "mathtext.fontset": "cm",
    "font.size": 18,

    "axes.titlesize": 21,
    "axes.labelsize": 19,

    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    "legend.fontsize": 16,

    "figure.titlesize": 21,

    # Vector output with embedded TrueType fonts
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# PAPER FIGURE STYLE (Figures 7 and 8)
#
# The paper figures are drawn at their final printed size
# (\textwidth = 472 pt = 6.53 in), so the font sizes below are
# the sizes that appear in the PDF. They are applied only inside
# `paper_style()` and do not affect the other figures.
# ============================================================

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
    "savefig.dpi": 400,
}

# Okabe-Ito colour-blind safe palette
APPROACH_COLORS = {
    "Light Only": "#0072B2",
    "Best Only": "#009E73",
    "Two-Step": "#E69F00",
}

# Shading for the premature-alarm region (latency < 0)
PREMATURE_COLOR = "#D55E00"


def paper_style():
    """Context manager with the paper figure style."""
    return mpl.rc_context(PAPER_RC)


def save_paper_figure(fig, filename):
    """
    Save a paper figure as vector PDF (for LaTeX) and PNG (preview).
    `filename` may be given with or without extension.
    """
    stem = os.path.splitext(filename)[0]
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{stem}.{ext}",
            bbox_inches="tight",
            pad_inches=0.02
        )
    print(f"  Saved {stem}.pdf and {stem}.png")

# ============================================================
# APPROACH MAPPING
#
# The configurations (IDs, parameters and the workbook/sheet of
# their runs) are defined once in configs.py. The loaded data are
# keyed by configuration ID, so APPROACH_MAPPING is the identity
# (kept so that the functions below can use it unchanged).
# ============================================================

from configurations import CONFIGS, RESULT_FILES, SHEET_TO_ID, fmt_conf, rule_label
from configurations import ids as config_ids
from configurations import label as config_label

APPROACH_MAPPING = {cid: cid for cid in CONFIGS}

APPROACH_TO_ID = dict(APPROACH_MAPPING)

FIGURE_LABELS = {cid: config_label(cid) for cid in CONFIGS}

# Description of every configuration, used by the paper figures
# (approach, (W,T) rule, light confidence, heavy confidence).
# The (W,T) notation follows the paper: an event is positive when
# at least T of the last W processed frames contain a detection.
CONFIG_INFO = {
    cid: (
        c["approach"],
        rule_label(c["rule"]) if c["rule"] else "",
        c["c_L"],
        c["c_H"],
    )
    for cid, c in CONFIGS.items()
}


def config_group(approach_id):
    """
    Group key and group label of a configuration. The approach is
    encoded by colour (legend), so the group label only gives the
    confidence thresholds:
        Light Only : c_L
        Two-Step   : c_L and c_H
        Best Only  : all Best Only boxes form one group; the tick
                     labels give c_H
    """
    approach, _, c_light, c_heavy = CONFIG_INFO[approach_id]

    if approach == "Light Only":
        return (approach, c_light), rf"$c_L$ = {fmt_conf(c_light)}"

    if approach == "Best Only":
        return (approach,), r"$c_H$"

    return (
        (approach, c_light, c_heavy),
        rf"$c_L$ = {fmt_conf(c_light)}" + "\n" + rf"$c_H$ = {fmt_conf(c_heavy)}"
    )


def config_tick_label(approach_id):
    """Short tick label: the (W,T) rule, or c_H for Best Only."""
    approach, rule, _, c_heavy = CONFIG_INFO[approach_id]
    return rule if approach != "Best Only" else fmt_conf(c_heavy)


def approach_legend(ax, approaches, y=1.13):
    """Colour legend of the approaches, above the axes."""
    handles = [
        Patch(
            facecolor=mpl.colors.to_rgba(APPROACH_COLORS[a], 0.55),
            edgecolor=APPROACH_COLORS[a],
            linewidth=0.6,
            label=a
        )
        for a in approaches
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.2,
        borderaxespad=0
    )


def draw_group_labels(ax, ids, y_offset=-0.11, fontsize=7, line_offset=0.035):
    """
    Write the group label centred under each run of consecutive
    configurations with the same group, with a thin bracket line,
    and return the positions after which a group ends.
    """
    trans = mpl.transforms.blended_transform_factory(
        ax.transData,
        ax.transAxes
    )

    groups = []
    for position, approach_id in enumerate(ids, start=1):
        key, label = config_group(approach_id)
        if groups and groups[-1]["key"] == key:
            groups[-1]["end"] = position
        else:
            groups.append({
                "key": key,
                "label": label,
                "start": position,
                "end": position,
            })

    for group in groups:
        x0, x1 = group["start"] - 0.4, group["end"] + 0.4
        ax.plot(
            [x0, x1],
            [y_offset + line_offset] * 2,
            transform=trans,
            color="0.35",
            linewidth=0.6,
            clip_on=False
        )
        ax.text(
            (group["start"] + group["end"]) / 2,
            y_offset,
            group["label"],
            transform=trans,
            ha="center",
            va="top",
            fontsize=fontsize,
            linespacing=1.15
        )

    return [group["end"] for group in groups[:-1]]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def natural_sort_key(text):
    """
    Natural sorting.

    Example:
        video1
        video2
        video10

    instead of:
        video1
        video10
        video2
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def add_latency_minutes_yaxis(ax):
    """
    Add secondary Y axis converting seconds to minutes.
    """

    sec_to_min = lambda x: x / 60.0
    min_to_sec = lambda x: x * 60.0

    ax_min = ax.secondary_yaxis(
        "right",
        functions=(sec_to_min, min_to_sec)
    )

    ax_min.set_ylabel("Latency (min)")

    return ax_min


def add_latency_minutes_xaxis(ax):
    """
    Add secondary X axis converting seconds to minutes.
    """

    sec_to_min = lambda x: x / 60.0
    min_to_sec = lambda x: x * 60.0

    ax_min = ax.secondary_xaxis(
        "top",
        functions=(sec_to_min, min_to_sec)
    )

    ax_min.set_xlabel("Latency (min)")

    return ax_min

def add_false_alarm_region_y(ax):
    """
    Fill the entire existing axes area below y=0.

    IMPORTANT:
    - Does NOT change Y-axis limits.
    - Does NOT change X-axis limits.
    - Uses the current limits exactly as they are.
    - Fills the complete width of the plot below y=0.
    """

    ymin, ymax = ax.get_ylim()

    # Zero is not inside the current Y-axis.
    if ymin >= 0 or ymax <= 0:
        return

    # Position of y=0 in axes coordinates.
    y_zero = (0.0 - ymin) / (ymax - ymin)

    red_region = Rectangle(
        (0, 0),
        1,
        y_zero,
        transform=ax.transAxes,
        facecolor="red",
        alpha=0.20,
        edgecolor="none",
        zorder=-10
    )

    ax.add_patch(red_region)

    # Zero boundary
    ax.axhline(
        0,
        linewidth=1.2,
        zorder=5
    )


def add_false_alarm_region_x(ax):
    """
    Fill the entire existing axes area to the left of x=0.

    IMPORTANT:
    - Does NOT change X-axis limits.
    - Does NOT change Y-axis limits.
    - Uses the current limits exactly as they are.
    - Fills the complete height of the plot to the left of x=0.
    """

    xmin, xmax = ax.get_xlim()

    # Zero is not inside the current X-axis.
    if xmin >= 0 or xmax <= 0:
        return

    # Position of x=0 in axes coordinates.
    x_zero = (0.0 - xmin) / (xmax - xmin)

    red_region = Rectangle(
        (0, 0),
        x_zero,
        1,
        transform=ax.transAxes,
        facecolor="red",
        alpha=0.20,
        edgecolor="none",
        zorder=-10
    )

    ax.add_patch(red_region)

    # Zero boundary
    ax.axvline(
        0,
        linewidth=1.2,
        zorder=5
    )

# ============================================================
# LOAD ALL RESULTS
# ============================================================

def load_all_results():
    """
    Read the runs of every configuration in configurations.CONFIGS.

    Returns {configuration ID: {video: [latency of each run]}}.
    The workbooks of each group are listed in configurations.RESULT_FILES;
    only the final results*.xlsx are read (partial_results*.xlsx are
    intermediate copies of the same runs). A sheet is identified by
    its group and name, since the same truncated sheet name means a
    different configuration in different groups.
    """

    data = {}

    for group, files in RESULT_FILES.items():

        if not files:
            print(f"\nNo workbooks for group '{group}'.")
            continue

        for filepath in files:

            print(f"\nReading ({group}): {filepath.name}")

            xls = pd.ExcelFile(filepath)

            for sheet_name in xls.sheet_names:

                approach_id = SHEET_TO_ID.get((group, sheet_name))

                if approach_id is None:
                    print(
                        f"  Skipping unknown approach/sheet: "
                        f"{sheet_name}"
                    )
                    continue

                df = pd.read_excel(xls, sheet_name=sheet_name)

                data.setdefault(approach_id, {})

                # Each column = video ("run" is the run index, not a video)
                for video in df.columns:

                    if str(video).strip().lower() == "run":
                        continue

                    values = pd.to_numeric(
                        df[video],
                        errors="coerce"
                    ).dropna()

                    if len(values) == 0:
                        continue

                    data[approach_id].setdefault(video, []).extend(
                        values.tolist()
                    )

    missing = [cid for cid in CONFIGS if cid not in data]
    if missing:
        print(f"\nWARNING: no results for {missing}")

    return data


# ============================================================
# VIDEO MAPPING
# ============================================================

def create_video_mapping(data):

    all_videos = set()

    for approach_data in data.values():

        all_videos.update(
            approach_data.keys()
        )

    all_videos = sorted(
        all_videos,
        key=natural_sort_key
    )

    video_mapping = {
        i + 1: video
        for i, video in enumerate(all_videos)
    }

    return video_mapping


# ============================================================
# CREATE STATISTICS
# ============================================================

def create_statistics(
    data,
    video_mapping
):

    approach_ids = list(
        APPROACH_MAPPING.keys()
    )

    video_ids = list(
        video_mapping.keys()
    )

    mean_df = pd.DataFrame(
        index=video_ids,
        columns=approach_ids,
        dtype=float
    )

    median_df = pd.DataFrame(
        index=video_ids,
        columns=approach_ids,
        dtype=float
    )

    std_df = pd.DataFrame(
        index=video_ids,
        columns=approach_ids,
        dtype=float
    )

    for video_id, video_name in video_mapping.items():

        for approach_id in approach_ids:

            approach = APPROACH_MAPPING[
                approach_id
            ]

            values = data.get(
                approach,
                {}
            ).get(
                video_name,
                []
            )

            if len(values) == 0:
                continue

            series = pd.Series(
                values,
                dtype=float
            )

            # Mean
            mean_df.loc[
                video_id,
                approach_id
            ] = series.mean()

            # Median
            median_df.loc[
                video_id,
                approach_id
            ] = series.median()

            # Sample standard deviation
            std_df.loc[
                video_id,
                approach_id
            ] = series.std()

    return mean_df, median_df, std_df


# ============================================================
# MEAN ± STD DISPLAY
# ============================================================

def create_mean_std_display(
    mean_df,
    std_df
):

    display_df = (
        mean_df
        .copy()
        .astype(object)
    )

    for row in mean_df.index:

        for col in mean_df.columns:

            mean_value = mean_df.loc[
                row,
                col
            ]

            std_value = std_df.loc[
                row,
                col
            ]

            if pd.isna(mean_value):

                display_df.loc[
                    row,
                    col
                ] = ""

            elif pd.isna(std_value):

                display_df.loc[
                    row,
                    col
                ] = f"{mean_value:.2f} s"

            else:

                display_df.loc[
                    row,
                    col
                ] = (
                    f"{mean_value:.2f} ± "
                    f"{std_value:.2f} s"
                )

    return display_df


# ============================================================
# IMPROVEMENT TABLE
# ============================================================

def create_improvement_table(mean_df):

    improvement_df = pd.DataFrame(
        index=mean_df.index,
        columns=mean_df.columns,
        dtype=float
    )

    if BASELINE_APPROACH not in APPROACH_TO_ID:

        raise ValueError(
            f"Baseline approach "
            f"'{BASELINE_APPROACH}' "
            f"is not present in APPROACH_MAPPING."
        )

    baseline_id = APPROACH_TO_ID[
        BASELINE_APPROACH
    ]

    for video_id in mean_df.index:

        baseline = mean_df.loc[
            video_id,
            baseline_id
        ]

        if (
            pd.isna(baseline)
            or baseline == 0
        ):
            continue

        for approach_id in mean_df.columns:

            value = mean_df.loc[
                video_id,
                approach_id
            ]

            if pd.isna(value):
                continue

            improvement_df.loc[
                video_id,
                approach_id
            ] = (
                (baseline - value)
                / baseline
                * 100
            )

    return improvement_df


# ============================================================
# SAVE EXCEL
# ============================================================

def save_excel(
    mean_df,
    median_df,
    std_df,
    mean_std_df,
    improvement_df,
    video_mapping
):

    print(
        f"\nWriting Excel file: {OUTPUT_EXCEL}"
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Approach Mapping
        # ----------------------------------------------------

        mapping_rows = []

        for short_id, approach in (
            APPROACH_MAPPING.items()
        ):

            is_baseline = (
                approach == BASELINE_APPROACH
            )

            mapping_rows.append({
                "Approach ID": short_id,
                "Approach": FIGURE_LABELS[short_id],
                "Workbook group": CONFIGS[short_id]["group"],
                "Sheet": CONFIGS[short_id]["sheet"],
                "Baseline": (
                    "YES"
                    if is_baseline
                    else ""
                )
            })

        approach_mapping_df = pd.DataFrame(
            mapping_rows
        )

        approach_mapping_df.to_excel(
            writer,
            sheet_name="Approach Mapping",
            index=False
        )

        # ----------------------------------------------------
        # Video Mapping
        # ----------------------------------------------------

        video_mapping_df = pd.DataFrame(
            [
                {
                    "Video ID": video_id,
                    "Video": video_name
                }
                for video_id, video_name
                in video_mapping.items()
            ]
        )

        video_mapping_df.to_excel(
            writer,
            sheet_name="Video Mapping",
            index=False
        )

        # ----------------------------------------------------
        # Mean ± STD
        # ----------------------------------------------------

        mean_std_output = mean_std_df.copy()

        mean_std_output.index.name = "Video ID"

        mean_std_output.to_excel(
            writer,
            sheet_name="Mean ± STD (s)"
        )

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        mean_output = mean_df.copy()

        mean_output.index.name = "Video ID"

        mean_output.to_excel(
            writer,
            sheet_name="Mean (s)"
        )

        # ----------------------------------------------------
        # Median
        # ----------------------------------------------------

        median_output = median_df.copy()

        median_output.index.name = "Video ID"

        median_output.to_excel(
            writer,
            sheet_name="Median (s)"
        )

        # ----------------------------------------------------
        # STD
        # ----------------------------------------------------

        std_output = std_df.copy()

        std_output.index.name = "Video ID"

        std_output.to_excel(
            writer,
            sheet_name="STD (s)"
        )

        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        improvement_output = (
            improvement_df
            .copy()
            .round(2)
        )

        improvement_output.index.name = "Video ID"

        improvement_output.to_excel(
            writer,
            sheet_name="Improvement (%)"
        )

        # ----------------------------------------------------
        # Excel formatting
        # ----------------------------------------------------

        workbook = writer.book

        for worksheet in workbook.worksheets:

            for column_cells in worksheet.columns:

                max_length = 0

                for cell in column_cells:

                    try:

                        value_length = len(
                            str(cell.value)
                        )

                        max_length = max(
                            max_length,
                            value_length
                        )

                    except Exception:
                        pass

                worksheet.column_dimensions[
                    column_cells[
                        0
                    ].column_letter
                ].width = min(
                    max_length + 2,
                    60
                )

            for cell in worksheet[1]:

                cell.font = Font(
                    bold=True
                )

                cell.alignment = Alignment(
                    horizontal="center"
                )

    print("Excel saved.")


# ============================================================
# FIGURE 1
# OVERALL BOXPLOTS
# ============================================================
def figure_01_overall_boxplots(data):
    """
    Overall latency boxplots of all configurations (paper Figure 12),
    in three rows sharing the latency axis:

        Light Only (c_L = 0.125 ... 0.75) || Best Only
        Two-Step with c_L = 0.125
        Two-Step with c_L = 0.25, 0.50 and 0.75
    """

    print(
        "\nGenerating Figure 1: "
        "Overall boxplots..."
    )

    figure_07_selected_boxplots(
        data=data,
        selected_ids=[
            config_ids("Light Only") + config_ids("Best Only"),
            config_ids("Two-Step", c_L=0.125),
            [i for c in (0.25, 0.50, 0.75) for i in config_ids("Two-Step", c_L=c)],
        ],
        filename="latency_all_configurations",
        section_breaks=[[len(config_ids("Light Only"))], [], []],
        width_fraction=1.0,
        height_in=2.15
    )


# ============================================================
# FIGURE 2
# MEAN LATENCY HEATMAP
# ============================================================

def figure_02_mean_latency_heatmap(mean_df):

    print(
        "\nGenerating Figure 2: "
        "Mean latency heatmap..."
    )

    fig_height = max(
        8,
        len(mean_df) * 0.22
    )

    fig, ax = plt.subplots(
        figsize=(16, fig_height)
    )

    matrix = mean_df.values.astype(float)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xticks(
        range(len(mean_df.columns))
    )

    ax.set_xticklabels(
        mean_df.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(mean_df.index))
    )

    ax.set_yticklabels(
        mean_df.index
    )

    ax.set_xlabel(
        "Approach"
    )

    ax.set_ylabel(
        "Video ID"
    )

    ax.set_title(
        "Mean Latency per Video and Approach"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Mean Latency (s)"
    )

    plt.tight_layout()

    plt.savefig(
        "figure_02_mean_latency_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 3
# RELATIVE IMPROVEMENT HEATMAP
# ============================================================

def figure_03_improvement_heatmap(
    improvement_df
):

    print(
        "\nGenerating Figure 3: "
        "Relative improvement heatmap..."
    )

    fig_height = max(
        8,
        len(improvement_df) * 0.22
    )

    fig, ax = plt.subplots(
        figsize=(16, fig_height)
    )

    matrix = improvement_df.values.astype(float)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_xticks(
        range(len(improvement_df.columns))
    )

    ax.set_xticklabels(
        improvement_df.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(improvement_df.index))
    )

    ax.set_yticklabels(
        improvement_df.index
    )

    ax.set_xlabel(
        "Approach"
    )

    ax.set_ylabel(
        "Video ID"
    )

    ax.set_title(
        "Latency Improvement Relative to L1 (%)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Improvement (%)"
    )

    plt.tight_layout()

    plt.savefig(
        "figure_03_relative_improvement_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 4
# MEAN ± STD
# ============================================================

def figure_04_mean_std(
    mean_df,
    std_df
):

    print(
        "\nGenerating Figure 4: "
        "Mean ± STD..."
    )

    approach_ids = list(
        mean_df.columns
    )

    means = []
    stds = []
    labels = []

    for approach_id in approach_ids:

        values = (
            mean_df[approach_id]
            .dropna()
        )

        std_values = (
            std_df[approach_id]
            .dropna()
        )

        if len(values) == 0:
            continue

        means.append(
            values.mean()
        )

        stds.append(
            std_values.mean()
        )

        labels.append(
            approach_id
        )

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=4
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_xlabel(
        "Approach"
    )

    ax.set_ylabel(
        "Latency (s)"
    )

    add_latency_minutes_yaxis(ax)

    ax.set_title(
        "Mean Latency ± Mean Standard Deviation"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "figure_04_mean_std.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 5
# CDF
# ============================================================

def figure_05_cdf(data):

    print(
        "\nGenerating Figure 5: "
        "CDF..."
    )

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for approach_id, approach in (
        APPROACH_MAPPING.items()
    ):

        values = []

        for video_values in data.get(
            approach,
            {}
        ).values():

            values.extend(
                video_values
            )

        if len(values) == 0:
            continue

        values = np.sort(
            np.asarray(values)
        )

        y = (
            np.arange(
                1,
                len(values) + 1
            )
            / len(values)
        )

        ax.plot(
            values,
            y,
            label=approach_id
        )

    add_false_alarm_region_x(ax)

    ax.set_xlabel(
        "Latency (s)"
    )

    add_latency_minutes_xaxis(ax)

    ax.set_ylabel(
        "CDF"
    )

    ax.set_title(
        "Latency CDF by Approach"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="best",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "figure_05_cdf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 6
# VIDEO × APPROACH BOXPLOT GRIDS
# ============================================================

def figure_06_boxplot_grids(
    data,
    video_mapping
):

    print(
        "\nGenerating Figure 6: "
        "Video × Approach boxplot grids..."
    )

    approach_ids = list(
        APPROACH_MAPPING.keys()
    )

    video_ids = list(
        video_mapping.keys()
    )

    # --------------------------------------------------------
    # Find videos with at least one result
    # --------------------------------------------------------

    valid_video_ids = []

    for video_id in video_ids:

        video_name = video_mapping[
            video_id
        ]

        has_data = False

        for approach in (
            APPROACH_MAPPING.values()
        ):

            values = data.get(
                approach,
                {}
            ).get(
                video_name,
                []
            )

            if len(values) > 0:

                has_data = True
                break

        if has_data:

            valid_video_ids.append(
                video_id
            )

    if not valid_video_ids:
        return

    # --------------------------------------------------------
    # Split videos across N_FIGURES
    # --------------------------------------------------------

    N = max(
        1,
        min(
            N_FIGURES,
            len(valid_video_ids)
        )
    )

    chunks = np.array_split(
        valid_video_ids,
        N
    )

    for figure_number, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk = list(chunk)

        if not chunk:
            continue

        n_rows = len(chunk)
        n_cols = len(approach_ids)

        fig_width = max(
            16,
            n_cols * 2.2
        )

        fig_height = max(
            8,
            n_rows * 2.5
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                fig_width,
                fig_height
            ),
            squeeze=False
        )

        # ----------------------------------------------------
        # Rows = videos
        # Columns = approaches
        # ----------------------------------------------------

        for row, video_id in enumerate(chunk):

            video_name = video_mapping[
                video_id
            ]

            for col, approach_id in enumerate(
                approach_ids
            ):

                ax = axes[
                    row,
                    col
                ]

                approach = APPROACH_MAPPING[
                    approach_id
                ]

                values = data.get(
                    approach,
                    {}
                ).get(
                    video_name,
                    []
                )

                # ------------------------------------------------
                # Boxplot
                # ------------------------------------------------

                if len(values) > 0:

                    ax.boxplot(
                        values,
                        showfliers=True
                    )

                # ------------------------------------------------
                # Title
                # ------------------------------------------------

                ax.set_title(
                    approach_id,
                    fontsize=9
                )

                # ------------------------------------------------
                # Y axis
                # ------------------------------------------------

                if col == 0:

                    ax.set_ylabel(
                        f"Video {video_id}\n"
                        f"Latency (s)"
                    )

                    add_latency_minutes_yaxis(
                        ax
                    )

                else:

                    ax.set_ylabel("")

                # ------------------------------------------------
                # Red region below 0
                #
                # No FALSE ALARMS text
                # ------------------------------------------------

                add_false_alarm_region_y(
                    ax
                )

                # ------------------------------------------------
                # Remove x ticks
                # ------------------------------------------------

                ax.set_xticks([])

                # ------------------------------------------------
                # Grid
                # ------------------------------------------------

                ax.grid(
                    axis="y",
                    alpha=0.25
                )

        # --------------------------------------------------------
        # Figure title
        # --------------------------------------------------------

        fig.suptitle(
            (
                "Latency Distribution per "
                "Video and Approach\n"
                f"Videos {chunk[0]}–{chunk[-1]}"
            ),
            fontsize=16
        )

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97
            ]
        )

        filename = (
            f"figure_06_boxplot_grid_"
            f"{figure_number}.png"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"  Saved {filename}"
        )


# ============================================================
# PRINT MAPPINGS
# ============================================================

def print_mappings(video_mapping):

    print("\n")

    print("=" * 70)
    print("APPROACH MAPPING")
    print("=" * 70)

    for short_id, approach in (
        APPROACH_MAPPING.items()
    ):

        baseline_text = ""

        if approach == BASELINE_APPROACH:

            baseline_text = "  [BASELINE]"

        print(
            f"{short_id:>3} -> "
            f"{FIGURE_LABELS[short_id]:<22} "
            f"{CONFIGS[short_id]['group']}:{CONFIGS[short_id]['sheet']}"
            f"{baseline_text}"
        )

    print("\n")

    print("=" * 70)
    print("VIDEO MAPPING")
    print("=" * 70)

    for video_id, video_name in (
        video_mapping.items()
    ):

        print(
            f"{video_id:>3} -> "
            f"{video_name}"
        )

    print("=" * 70)



# ============================================================
# FIGURE 7
# SELECTED CONFIGURATION BOXPLOTS
#
# Order:
# B1 | T1 T2 T3 | B2 | T4 T5 T6 T7 T8 T9
# ============================================================
def _config_values(data, approach_id):
    """All finite observations (runs x videos) of a configuration."""
    values = []
    for video_values in data.get(APPROACH_MAPPING[approach_id], {}).values():
        values.extend(video_values)
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def figure_07_selected_boxplots(
    data,
    selected_ids,
    filename,
    section_breaks=(),
    width_fraction=1.0,
    height_in=2.6,
):
    """
    Latency boxplots of selected configurations (paper figures).

    - Box colour  : approach (Light Only / Best Only / Two-Step).
    - Tick label  : (W,T) rule, or c_H for Best Only.
    - Group label : confidence thresholds of each run of boxes.
    - Number above each box: rank by median latency over all the
      boxes of the figure (1 = fastest).
    - Shaded region below zero: premature alarms (latency < 0).

    selected_ids:
        A list of configuration IDs (one row of boxes) or a list of
        such lists (one row per list, sharing the latency axis).

    section_breaks:
        Box numbers after which a thicker separator is drawn; for
        several rows, one sequence per row.

    width_fraction:
        Fraction of the text width the figure occupies in the paper.

    height_in:
        Height of each row (inches).
    """

    print(
        f"\nGenerating Figure 7: {filename}..."
    )

    single_row = not (selected_ids and isinstance(selected_ids[0], (list, tuple)))
    rows = [list(selected_ids)] if single_row else [list(r) for r in selected_ids]
    if single_row:
        section_breaks = [list(section_breaks)]
    else:
        section_breaks = [list(b) for b in section_breaks]
        section_breaks += [[]] * (len(rows) - len(section_breaks))

    # ========================================================
    # Collect data (all observations of every configuration)
    # ========================================================

    row_ids, row_values = [], []

    for ids_ in rows:
        valid, vals = [], []
        for approach_id in ids_:
            if approach_id not in APPROACH_MAPPING:
                print(f"WARNING: Unknown approach ID: {approach_id}")
                continue
            values = _config_values(data, approach_id)
            if len(values) == 0:
                print(f"WARNING: No data for {approach_id}")
                continue
            valid.append(approach_id)
            vals.append(values)
        row_ids.append(valid)
        row_values.append(vals)

    flat_ids = [i for r in row_ids for i in r]
    flat_values = [v for r in row_values for v in r]

    if not flat_ids:
        print("No data available for Figure 7.")
        return

    # ========================================================
    # Rank configurations by median latency (1 = fastest)
    # ========================================================

    medians = np.array([np.median(v) for v in flat_values])
    sorted_indices = np.argsort(medians, kind="stable")
    rank_of = {}
    for rank, index in enumerate(sorted_indices, start=1):
        rank_of[flat_ids[index]] = rank

    print("\nMedian latency ranking:")
    for index in sorted_indices:
        print(
            f"  {rank_of[flat_ids[index]]:>2}. "
            f"{flat_ids[index]:>3} "
            f"({FIGURE_LABELS[flat_ids[index]]}) "
            f"median = {medians[index]:.2f} s"
        )

    data_min = min(np.min(v) for v in flat_values)
    data_max = max(np.max(v) for v in flat_values)
    span = data_max - min(data_min, 0)
    ymin = min(data_min, 0) - 0.05 * span
    ymax = data_max + 0.05 * span

    width_in = TEXT_WIDTH_IN * width_fraction
    approaches_in_fig = [CONFIG_INFO[i][0] for i in flat_ids]

    # ========================================================
    # Figure
    # ========================================================

    with paper_style():

        fig, axes = plt.subplots(
            len(row_ids), 1,
            figsize=(width_in, height_in * len(row_ids)),
            squeeze=False
        )
        axes = axes[:, 0]

        for row, (ax, valid_ids, all_values, breaks) in enumerate(
            zip(axes, row_ids, row_values, section_breaks)
        ):
            positions = np.arange(1, len(valid_ids) + 1)
            approaches = [CONFIG_INFO[i][0] for i in valid_ids]

            box = ax.boxplot(
                all_values,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="black", linewidth=1.0),
                whiskerprops=dict(color="0.25", linewidth=0.6),
                capprops=dict(color="0.25", linewidth=0.6),
                flierprops=dict(
                    marker="o",
                    markersize=2.2,
                    markerfacecolor="none",
                    markeredgecolor="0.45",
                    markeredgewidth=0.4
                ),
                zorder=3
            )

            for patch, approach in zip(box["boxes"], approaches):
                color = APPROACH_COLORS[approach]
                patch.set_facecolor(mpl.colors.to_rgba(color, 0.55))
                patch.set_edgecolor(color)
                patch.set_linewidth(0.7)

            # ------------------------------------------------
            # Axes limits and premature-alarm region
            # ------------------------------------------------

            ax.set_xlim(0.4, len(valid_ids) + 0.6)
            ax.set_ylim(ymin, ymax)

            if ymin < 0:
                ax.axhspan(ymin, 0, facecolor=PREMATURE_COLOR, alpha=0.10,
                           edgecolor="none", zorder=0)
                ax.axhline(0, color=PREMATURE_COLOR, linewidth=0.7, zorder=2)
                ax.text(0.995, 0.015, "Premature alarm", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, style="italic",
                        color=PREMATURE_COLOR, zorder=4)

            ax.grid(axis="y", zorder=0)
            ax.set_axisbelow(True)

            # ------------------------------------------------
            # Ticks, group labels and separators
            # ------------------------------------------------

            box_width_in = width_in / max(len(valid_ids), 1)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [config_tick_label(i) for i in valid_ids],
                fontsize=7 if box_width_in > 0.3 else 6.5
            )
            ax.tick_params(axis="x", length=0, pad=2)

            # offsets in inches (below the tick labels), whatever the row height
            boundaries = draw_group_labels(
                ax, valid_ids,
                y_offset=-0.30 / height_in,
                line_offset=0.06 / height_in,
                fontsize=7 if box_width_in > 0.3 else 6.5
            )

            for boundary in boundaries:
                thick = boundary in breaks
                ax.axvline(
                    boundary + 0.5,
                    color="0.2" if thick else "0.65",
                    linewidth=1.0 if thick else 0.5,
                    zorder=1
                )

            ax.set_ylabel("Latency (s)")

            ax_min = add_latency_minutes_yaxis(ax)
            ax_min.set_ylabel("Latency (min)")
            ax_min.tick_params(axis="y", labelsize=8)
            ax_min.spines["right"].set_linewidth(0.6)

            # ------------------------------------------------
            # Ranks above the boxes
            # ------------------------------------------------

            rank_trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

            for position, approach_id in zip(positions, valid_ids):
                ax.text(position, 1.015, str(rank_of[approach_id]), transform=rank_trans,
                        ha="center", va="bottom", fontsize=7 if box_width_in > 0.22 else 6,
                        color="0.2")

            ax.text(0.4, 1.015, "Rank", transform=rank_trans, ha="right", va="bottom",
                    fontsize=7, style="italic", color="0.2")

        approach_legend(
            axes[0],
            [a for a in APPROACH_COLORS if a in approaches_in_fig],
            y=1.09 if len(row_ids) == 1 else 1.12
        )

        if len(row_ids) > 1:
            # room for the two-line group labels between the rows
            fig.subplots_adjust(hspace=0.62)

        save_paper_figure(fig, filename)
        plt.close(fig)


def false_alarm_rate(data, approach_id):
    """Premature alarms: observations < 0, in % of all observations."""
    all_values = []
    for video_values in data.get(APPROACH_MAPPING[approach_id], {}).values():
        all_values.extend(video_values)
    if len(all_values) == 0:
        return np.nan
    all_values = np.asarray(all_values, dtype=float)
    return np.sum(all_values < 0) / len(all_values) * 100


def figure_08_false_alarm_rate(data, panels):
    """
    False-alarm-rate figures (paper Figure 13), one file per panel:
        panels = [(filename, [configuration IDs], width fraction), ...]

    False alarm:
        Any observation < 0 (alarm before the annotated onset).

    False Alarm Rate:
        number of negative observations / total observations * 100

    All panels share the y-axis limit; each is drawn at the width it
    occupies in the paper.
    """

    print(
        "\nGenerating Figure 8: False-alarm rate..."
    )

    rates = {
        approach_id: false_alarm_rate(data, approach_id)
        for _, ids_, _ in panels
        for approach_id in ids_
    }

    print("\n--- False Alarm Rate ---")

    for approach_id, rate in rates.items():
        print(
            f"  {approach_id} ({FIGURE_LABELS[approach_id]}): "
            f"{rate:.2f}%"
        )

    # Same y-axis limit for all panels
    y_max = np.nanmax(list(rates.values())) * 1.18

    for filename, ids_, width_fraction in panels:

        ids_ = [i for i in ids_ if not np.isnan(rates[i])]
        positions = np.arange(1, len(ids_) + 1)
        bar_width_in = TEXT_WIDTH_IN * width_fraction / max(len(ids_), 1)

        with paper_style():

            fig, ax = plt.subplots(
                figsize=(TEXT_WIDTH_IN * width_fraction, 1.85)
            )

            for position, approach_id in zip(positions, ids_):

                color = APPROACH_COLORS[CONFIG_INFO[approach_id][0]]
                rate = rates[approach_id]

                ax.bar(
                    position,
                    rate,
                    width=0.68,
                    facecolor=mpl.colors.to_rgba(color, 0.55),
                    edgecolor=color,
                    linewidth=0.7,
                    zorder=3
                )

                ax.text(
                    position,
                    rate + y_max * 0.015,
                    f"{rate:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6 if bar_width_in > 0.2 else 5,
                    rotation=0 if bar_width_in > 0.2 else 90,
                    zorder=4
                )

            ax.set_xlim(0.4, len(ids_) + 0.6)
            ax.set_ylim(0, y_max)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", zorder=0)
            ax.set_axisbelow(True)

            # Vertical tick labels: horizontal (W,T) labels would touch.
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [config_tick_label(i) for i in ids_],
                fontsize=6.5,
                rotation=90
            )
            ax.tick_params(axis="x", length=0, pad=2)
            ax.tick_params(axis="y", labelsize=7)

            boundaries = draw_group_labels(
                ax,
                ids_,
                y_offset=-0.25,
                fontsize=6.5
            )

            for boundary in boundaries:
                ax.axvline(
                    boundary + 0.5,
                    color="0.65",
                    linewidth=0.5,
                    zorder=1
                )

            ax.set_ylabel("Premature alarm rate (%)", fontsize=8)

            save_paper_figure(fig, filename)
            plt.close(fig)


# ============================================================
# APPENDIX TABLES: PER-VIDEO MEDIAN LATENCY OF ALL CONFIGURATIONS
#
# One uniform set of tables with every configuration (Appendix A):
#   1. Light Only (c_L = 0.125 ... 0.75) and Best Only (c_H = 0.125 ... 0.75)
#   2. Two-Step with c_L = 0.125 (c_H = 0.125 ... 0.75)
#   3. Two-Step with c_L = 0.25, 0.50 and 0.75 (c_H >= c_L)
# each split into videos 1-23, 24-46 and 47-69, i.e. nine tables in
# median_latency_all_1.tex ... median_latency_all_9.tex (labels
# tab:latency_1 ... tab:latency_9).
# ============================================================

def _latency_table_groups():
    two_step_high = [(cl, ch) for cl in (0.25, 0.50, 0.75) for ch in (0.25, 0.50, 0.75) if ch >= cl]
    return [
        (
            "Light Only and Best Only",
            [(rf"\makecell{{Light Only\\$c_L={fmt_conf(c)}$}}", config_ids("Light Only", c_L=c))
             for c in (0.125, 0.25, 0.50, 0.75)]
            + [(r"\makecell{Best Only\\$c_H$}", config_ids("Best Only"))],
        ),
        (
            r"Two-Step with $c_L=0.125$",
            [(rf"\makecell{{$c_L={fmt_conf(0.125)}$\\$c_H={fmt_conf(ch)}$}}", config_ids("Two-Step", c_L=0.125, c_H=ch))
             for ch in (0.125, 0.25, 0.50, 0.75)],
        ),
        (
            r"Two-Step with $c_L\geq0.25$",
            [(rf"\makecell{{$c_L={fmt_conf(cl)}$\\$c_H={fmt_conf(ch)}$}}", config_ids("Two-Step", c_L=cl, c_H=ch))
             for cl, ch in two_step_high],
        ),
    ]


VIDEO_RANGES = [(1, 23), (24, 46), (47, 69)]


def write_latency_tables(median_df):

    # No alarm: shaded cell with a cross (pifont \ding{55}, xcolor[table]).
    def fmt(v):
        return r"\cellcolor{gray!25}\ding{55}" if pd.isna(v) else f"{v:.2f}"

    number = 1

    for description, blocks in _latency_table_groups():

        columns = [cid for _, block_ids in blocks for cid in block_ids]
        col_spec = "c|" + "|".join("r" * len(block_ids) for _, block_ids in blocks)
        head_1 = " & ".join(
            [r"\textbf{Video}"]
            + [
                rf"\multicolumn{{{len(block_ids)}}}{{c{'|' if k < len(blocks) - 1 else ''}}}{{\textbf{{{title}}}}}"
                for k, (title, block_ids) in enumerate(blocks)
            ]
        ) + r" \\"
        # second header row: (W,T) rule, or c_H for Best Only
        head_2 = " & ".join([""] + [config_tick_label(cid) for cid in columns]) + r" \\"

        for start, end in VIDEO_RANGES:

            lines = [
                "% Generated by analysis/latency_analysis.py -- do not edit by hand.",
                r"\begin{sidewaystable}[p]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\caption{\hl{Median latency (in seconds) for videos "
                + f"{start}--{end}: {description}. Negative values indicate alarms raised before the "
                r"annotated onset, and shaded cells marked with a cross indicate that no alarm was raised.}}",
                rf"\label{{tab:latency_{number}}}",
                rf"\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                head_1,
                head_2,
                r"\midrule",
            ]
            for video_id, row in median_df.loc[start:end, columns].iterrows():
                lines.append(f"{video_id} & " + " & ".join(fmt(v) for v in row.values) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}", ""]

            filename = f"median_latency_all_{number}.tex"
            with open(filename, "w", encoding="utf-8") as out:
                out.write("\n".join(lines))
            print(f"Saved LaTeX table: {filename}")
            number += 1


def main():

    print(
        "\n"
        "============================================================\n"
        " Results Analysis\n"
        "============================================================"
    )

    # Load all Excel files
    data = load_all_results()

    # Create video mapping
    video_mapping = create_video_mapping(
        data
    )

    # Print mappings
    print_mappings(
        video_mapping
    )

    # Calculate mean, median and STD
    mean_df, median_df, std_df = create_statistics(
        data,
        video_mapping
    )

    # Create Mean ± STD table
    mean_std_df = create_mean_std_display(
        mean_df,
        std_df
    )


    # Calculate improvement relative to L1
    improvement_df = create_improvement_table(
        mean_df
    )

    # Save Excel
    save_excel(
        mean_df,
        median_df,
        std_df,
        mean_std_df,
        improvement_df,
        video_mapping
    )

    # ============================================================
    # Save mean table as LaTeX
    # ============================================================

    mean_latex = mean_df.copy()

    # Optional: round values
    mean_latex = mean_latex.round(2)

    mean_latex.to_latex(
        "mean_latency.tex",
        index=True,
        float_format="%.2f",
        escape=True
    )

    print("Saved LaTeX table: mean_latency.tex")

    median_latex = median_df.copy().round(2)

    def fmt(v):
        return "NaN" if pd.isna(v) else f"{v:.2f}"

    confidence_025_025 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.25; Confidence$_{best}$ = 0.25}
        }
    }"""

    confidence_025_050 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.25; Confidence$_{best}$ = 0.50}
        }
    }"""

    confidence_050_050 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.50; Confidence$_{best}$ = 0.50}
        }
    }"""

    median_latex = median_df.copy().round(2)

    def fmt(v):
        return "NaN" if pd.isna(v) else f"{v:.2f}"

    confidence_025_025 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.25; Confidence$_{best}$ = 0.25}
        }
    }"""

    confidence_025_050 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.25; Confidence$_{best}$ = 0.50}
        }
    }"""

    confidence_050_050 = r"""\multirow{23}{*}{
        \rotatebox[origin=c]{-90}{
            \textbf{Confidence$_{light}$ = 0.50; Confidence$_{best}$ = 0.50}
        }
    }"""

    # Column order of the LaTeX header of Tables 8-10:
    #   c_L = 0.25, c_H = 0.25 : Light Only x3 | Best Only | Two-Step x3
    #   c_L = 0.25, c_H = 0.50 : Best Only | Two-Step x3
    #   c_L = 0.50, c_H = 0.50 : Light Only x3 | Two-Step x3
    # (the values must NOT be written in the APPROACH_MAPPING order)
    latex_column_order = [
        "L1", "L2", "L3", "B1", "T1", "T3", "T5",
        "B2", "T2", "T4", "T6",
        "L4", "L5", "L6", "T7", "T8", "T9",
    ]

    for table_num, (start, end) in enumerate([(1, 23), (24, 46), (47, 69)], start=1):

        lines = []

        for video_id, row in median_latex.loc[start:end, latex_column_order].iterrows():

            vals = [fmt(v) for v in row.values]

            if video_id == start:

                line = (
                    f"{video_id}\n\n"
                    f"&\n\n"
                    f"{confidence_025_025}\n\n"
                    f"&\n\n"
                    f"{' & '.join(vals[0:3])}\n\n"
                    f"&\n\n"
                    f"{vals[3]}\n\n"
                    f"&\n\n"
                    f"{' & '.join(vals[4:7])}\n\n"
                    f"&\n\n"
                    f"{confidence_025_050}\n\n"
                    f"&\n\n"
                    f"{vals[7]}\n\n"
                    f"&\n\n"
                    f"{' & '.join(vals[8:11])}\n\n"
                    f"&\n\n"
                    f"{confidence_050_050}\n\n"
                    f"&\n\n"
                    f"{' & '.join(vals[11:14])}\n\n"
                    f"&\n\n"
                    f"{' & '.join(vals[14:17])}\n\n"
                    f"\\\\"
                )

            else:

                line = (
                    f"{video_id} & & "
                    f"{' & '.join(vals[:7])} & & "
                    f"{' & '.join(vals[7:11])} & & "
                    f"{' & '.join(vals[11:14])} & "
                    f"{' & '.join(vals[14:])} \\\\"
                )

            lines.append(line)

        with open(f"median_{table_num}.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


    write_latency_tables(median_df)

    # Generate figures
    figure_01_overall_boxplots(
        data
    )

    figure_02_mean_latency_heatmap(
        mean_df
    )

    figure_03_improvement_heatmap(
        improvement_df
    )

    figure_04_mean_std(
        mean_df,
        std_df
    )

    figure_05_cdf(
        data
    )

    #figure_06_boxplot_grids(
    #    data,
    #    video_mapping
    #)

    LO, BO, TS = (config_ids(a) for a in ("Light Only", "Best Only", "Two-Step"))

    def ts(c_L):
        return config_ids("Two-Step", c_L=c_L)

    def lo(c_L):
        return config_ids("Light Only", c_L=c_L)

    # ============================================================
    # FIGURE 7a  (paper Figure 16a: Best Only vs Two-Step, full width)
    #
    # Best Only || TS c_L=0.125 (c_H = 0.125 ... 0.75)
    # TS c_L=0.25 | TS c_L=0.50 | TS c_L=0.75
    # ============================================================

    figure_07_selected_boxplots(
        data=data,
        selected_ids=[BO + ts(0.125), ts(0.25) + ts(0.50) + ts(0.75)],
        filename="latency_best_only_vs_two_step",
        section_breaks=[[len(BO)], [len(ts(0.25)), len(ts(0.25)) + len(ts(0.50))]],
        width_fraction=1.0,
        height_in=2.3
    )

    # ============================================================
    # FIGURE 7b  (paper Figure 14: Light Only vs Two-Step, full width)
    #
    # One block per c_L: Light Only followed by the Two-Step
    # configurations with the same c_L (thick separators between
    # blocks).
    # ============================================================

    blocks = [lo(c) + ts(c) for c in (0.125, 0.25, 0.50, 0.75)]

    figure_07_selected_boxplots(
        data=data,
        selected_ids=[blocks[0], blocks[1], blocks[2] + blocks[3]],
        filename="latency_light_only_vs_two_step",
        section_breaks=[[], [], [len(blocks[2])]],
        width_fraction=1.0,
        height_in=2.3
    )

    # ============================================================
    # FIGURE 7c  (paper Figure 15a: Light Only vs Best Only, full width)
    #
    # LO c_L=0.125 | 0.25 | 0.50 | 0.75 || Best Only
    # ============================================================

    figure_07_selected_boxplots(
        data=data,
        selected_ids=LO + BO,
        filename="latency_light_only_vs_best_only",
        section_breaks=[len(LO)],
        width_fraction=1.0,
        height_in=2.6
    )

    # ============================================================
    # FIGURE 8  (paper Figure 13: premature-alarm rate)
    #   8a Light Only (0.62 of the text width), 8c Best Only (0.32),
    #   8b Two-Step (full width)
    # ============================================================

    figure_08_false_alarm_rate(data, [
        ("premature_alarm_rate_light_only", LO, 0.62),
        ("premature_alarm_rate_two_step", TS, 1.0),
        ("premature_alarm_rate_best_only", BO, 0.32),
    ])

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Analysis completed successfully.")
    print(f"Excel output: {OUTPUT_EXCEL}")

    print("\nGenerated figures:")

    print(
        "  latency_all_configurations.png"
    )

    print(
        "  figure_02_mean_latency_heatmap.png"
    )

    print(
        "  figure_03_relative_improvement_heatmap.png"
    )

    print(
        "  figure_04_mean_std.png"
    )

    print(
        "  figure_05_cdf.png"
    )

    """
    for i in range(
        1,
        N_FIGURES + 1
    ):

        print(
            f"  figure_06_boxplot_grid_{i}.png"
        )
    """

    print(
        "  figure_07_selected_boxplots.png"
    )

    print(
            "  premature_alarm_rate_light_only.png"
        )

    print(
            "  premature_alarm_rate_two_step.png"
        )

    print("=" * 70)




# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()