"""
Regenerate every result figure and table of the system-level evaluation and
copy the ones used in the paper to paper/figs and paper/tables.

    cd analysis
    uv sync                      # once (or: pip install -r requirements.txt)
    uv run python make_all.py    # or: python make_all.py

Steps (each script can also be run on its own, from analysis/output/):
  1. latency_analysis.py      results_aggregated.xlsx, Figures 12, 13a-c, 14, 15a, 16a
                              and the per-video latency tables (Tables A.1-A.9)
  2. pairwise_comparison.py   comparison_results.xlsx and Figures 15b, 16b
  3. theory_vs_observed.py    ideal-case prediction vs. observed (values of Section 4.5.3)
     ideal_latency.py         Table 9 and Figure 16c (ideal models vs. Eq. (4) vs. observed), from
                              results/positive_videos/ideal_latency.csv (experiments/ideal_latency.py)
  4. false_alarm_tables.py    Table 8 and Table A.10 (negative videos), written to paper/tables
  5. summary_table.py         Table 10 (all configurations and methods), written to paper/tables
  6. timing_diagram.py        Figure 3 (Two-Step vs Best Only timing on the RPi5)

The configurations are defined in configurations.py and the results are read
from results/ (see results/README.md). The frame figures of Figure 17 are made
separately by alarm_frames/ (they need the videos and the models).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from configurations import OUTPUT, PAPER

HERE = Path(__file__).resolve().parent
FIGS = PAPER / "figs"

STEPS = [
    "latency_analysis.py",
    "pairwise_comparison.py",
    "theory_vs_observed.py",
    "ideal_latency.py",
    "false_alarm_tables.py",
    "summary_table.py",
    "timing_diagram.py",
]

# figures used in the paper (file name in analysis/output = name in paper/figs)
PAPER_FIGURES = [
    "latency_all_configurations",          # Figure 12
    "premature_alarm_rate_light_only",     # Figure 13a
    "premature_alarm_rate_best_only",      # Figure 13b
    "premature_alarm_rate_two_step",       # Figure 13c
    "latency_light_only_vs_two_step",      # Figure 14
    "latency_light_only_vs_best_only",     # Figure 15a
    "figures/pairwise_light_only_vs_best_only",   # Figure 15b
    "latency_best_only_vs_two_step",       # Figure 16a
    "figures/pairwise_best_only_vs_two_step",     # Figure 16b
    "ideal_vs_observed_latency",           # Figure 16c
    "timing_two_step_vs_best_only",        # Figure 3
]

# per-video median latency tables of all configurations (Appendix A, Tables A.1-A.9)
PAPER_TABLES = [f"median_latency_all_{k}.tex" for k in range(1, 10)]


def main():
    OUTPUT.mkdir(exist_ok=True)
    # the scripts print non-ASCII characters (arrows, ±); keep that working
    # when the output is redirected on Windows
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(HERE)}
    for script in STEPS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], cwd=OUTPUT, check=True, env=env)

    for name in PAPER_FIGURES:
        src = OUTPUT / f"{name}.pdf"
        shutil.copy2(src, FIGS / src.name)
    print(f"\nCopied {len(PAPER_FIGURES)} figures to {FIGS}")

    for name in PAPER_TABLES:
        shutil.copy2(OUTPUT / name, PAPER / "tables" / name)
    print(f"Copied {len(PAPER_TABLES)} tables to {PAPER / 'tables'}")


if __name__ == "__main__":
    main()
