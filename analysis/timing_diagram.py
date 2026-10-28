"""
Figure 3 of the paper: timing of the Two-Step and Best Only approaches on the RPi5.

Illustrative scenario: the event becomes visible shortly after a high-capacity
(YOLOv12m) inference has started, so Best Only only sees it in the following
inference, whereas the lightweight model (YOLOv8n) samples the stream densely
and escalates to the high-capacity model earlier.

Every position and latency in the figure is computed from the parameters
below (inference times of Table 6 of the paper), so the figure can be
regenerated for other values. Output: timing_two_step_vs_best_only.{pdf,svg,png}
"""

import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle


# ============================================================
# PARAMETERS
# ============================================================

T_LIGHT = 0.184    # s per YOLOv8n inference on the RPi5 (Table 6)
T_HEAVY = 1.371    # s per YOLOv12m inference on the RPi5 (Table 6)
ONSET = 1.45       # s, instant at which the event becomes visible
RULE = (1, 1)      # (W,T) rule of the first stage (escalate on T of W frames)

TEXT_WIDTH_IN = 6.53   # \textwidth of the paper (472 pt)
OUTPUT = "timing_two_step_vs_best_only"


# ============================================================
# FONT (Latin Modern, as in the paper)
# ============================================================

LM_DIR = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Programs" / "MiKTeX" / "fonts" / "opentype" / "public" / "lm"
)

if (LM_DIR / "lmroman10-regular.otf").exists():
    for name in ("regular", "bold", "italic"):
        font_manager.fontManager.addfont(str(LM_DIR / f"lmroman10-{name}.otf"))
    FONT = font_manager.FontProperties(
        fname=str(LM_DIR / "lmroman10-regular.otf")
    ).get_name()
else:
    print(f"WARNING: Latin Modern not found in {LM_DIR}; using the default serif font.")
    FONT = "serif"

mpl.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# Same colours as the other figures (Okabe-Ito):
# lightweight model = Light Only blue, high-capacity model = Best Only green
LIGHT_COLOR = "#0072B2"
HEAVY_COLOR = "#009E73"
EVENT_COLOR = "#D55E00"
INK = "0.15"


# ============================================================
# TIMELINES
# ============================================================

def two_step_timeline():
    """
    Lightweight inferences run back-to-back; each analyses the frame available
    when it starts. With the (W,T) rule, escalation happens after the T-th
    positive frame (the event is assumed to be detected in every frame after
    the onset). The high-capacity model then verifies one frame and raises
    the alarm.
    """
    first = math.ceil(ONSET / T_LIGHT)            # first frame showing the event
    last = first + RULE[1] - 1                    # T-th positive frame
    light = [(k * T_LIGHT, k >= first) for k in range(last + 1)]
    escalation = (last + 1) * T_LIGHT
    alarm = escalation + T_HEAVY
    return light, escalation, alarm


def best_only_timeline():
    """High-capacity inferences back-to-back; alarm after the first positive frame."""
    first = math.ceil(ONSET / T_HEAVY)
    heavy = [(j * T_HEAVY, j >= first) for j in range(first + 1)]
    alarm = (first + 1) * T_HEAVY
    return heavy, alarm


light, escalation, alarm_2s = two_step_timeline()
heavy_bo, alarm_bo = best_only_timeline()
latency_2s = alarm_2s - ONSET
latency_bo = alarm_bo - ONSET

print(f"Two-Step : escalation {escalation:.3f} s, alarm {alarm_2s:.3f} s, latency {latency_2s:.3f} s")
print(f"Best Only: alarm {alarm_bo:.3f} s, latency {latency_bo:.3f} s")


# ============================================================
# FIGURE
# ============================================================

Y_2S, Y_BO = 1.0, 0.0      # lane centres
H = 0.40                   # block height
X_MAX = max(alarm_2s, alarm_bo) + 0.25

fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.35))
ax.set_xlim(0, X_MAX)
ax.set_ylim(-0.62, 1.75)

for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.set_yticks([])
ax.set_xticks([t / 2 for t in range(0, int(X_MAX * 2) + 1)])
ax.set_xticklabels([f"{t / 2:g}" if t % 2 == 0 else "" for t in range(0, int(X_MAX * 2) + 1)])
ax.set_xlabel("Time (s)", labelpad=2)


def block(x, y, width, color, positive, label=None):
    ax.add_patch(Rectangle(
        (x, y - H / 2), width, H,
        facecolor=mpl.colors.to_rgba(color, 0.75 if positive else 0.18),
        edgecolor=color,
        linewidth=0.6,
        zorder=2
    ))
    if label:
        ax.text(x + width / 2, y, label, ha="center", va="center",
                fontsize=7.5, color=INK, zorder=3)


def bracket(x0, x1, y, text, text_side="above"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK,
                                shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + (0.04 if text_side == "above" else -0.04), text,
            ha="center", va="bottom" if text_side == "above" else "top",
            fontsize=7.5, color=INK)


def alarm_marker(x, y):
    ax.plot([x, x], [y - H / 2 - 0.06, y + H / 2 + 0.06],
            color=INK, linewidth=1.3, zorder=4)
    ax.text(x, y + H / 2 + 0.08, "Alarm", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)


# ---------------- Two-Step lane ----------------
for x, positive in light:
    block(x, Y_2S, T_LIGHT, LIGHT_COLOR, positive)
block(escalation, Y_2S, T_HEAVY, HEAVY_COLOR, True, "verification")
alarm_marker(alarm_2s, Y_2S)

ax.text(escalation + 0.03, Y_2S + H / 2 + 0.08, "escalation", ha="left", va="bottom",
        fontsize=7.5, style="italic", color=INK)
bracket(0, T_LIGHT, Y_2S + H / 2 + 0.10, r"$t_L$ = " + f"{T_LIGHT:.3f} s")

# ---------------- Best Only lane ----------------
for i, (x, positive) in enumerate(heavy_bo):
    block(x, Y_BO, T_HEAVY, HEAVY_COLOR, positive,
          r"$t_H$ = " + f"{T_HEAVY:.3f} s" if i == 0 else None)
alarm_marker(alarm_bo, Y_BO)

# ---------------- Lane labels ----------------
for y, name in ((Y_2S, f"Two-Step\n$(W,T)=({RULE[0]},{RULE[1]})$"), (Y_BO, "Best Only")):
    ax.text(-0.06, y, name, ha="right", va="center", fontsize=8.5,
            linespacing=1.3, color=INK, clip_on=False)

# ---------------- Event onset and latencies ----------------
ax.plot([ONSET, ONSET], [Y_BO - H / 2 - 0.30, Y_2S + H / 2 + 0.36],
        color=EVENT_COLOR, linewidth=1.1, linestyle=(0, (4, 2)), zorder=5)
ax.text(ONSET, Y_2S + H / 2 + 0.38, "event onset", ha="center", va="bottom",
        fontsize=7.5, color=EVENT_COLOR)

bracket(ONSET, alarm_2s, Y_2S - H / 2 - 0.16, f"latency {latency_2s:.2f} s", "below")
bracket(ONSET, alarm_bo, Y_BO - H / 2 - 0.16, f"latency {latency_bo:.2f} s", "below")

# ---------------- Legend ----------------
ax.legend(
    handles=[
        Patch(facecolor=mpl.colors.to_rgba(LIGHT_COLOR, 0.18), edgecolor=LIGHT_COLOR,
              linewidth=0.6, label="lightweight model inference"),
        Patch(facecolor=mpl.colors.to_rgba(HEAVY_COLOR, 0.18), edgecolor=HEAVY_COLOR,
              linewidth=0.6, label="high-capacity model inference"),
        Patch(facecolor=mpl.colors.to_rgba("0.35", 0.75), edgecolor="0.35",
              linewidth=0.6, label="frame showing the event (darker)"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
    handlelength=1.4, columnspacing=1.6, borderaxespad=0.2
)


# ============================================================
# SAVE
# ============================================================

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{OUTPUT}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=400)
print(f"Saved {OUTPUT}.pdf/.svg/.png")
plt.close(fig)
