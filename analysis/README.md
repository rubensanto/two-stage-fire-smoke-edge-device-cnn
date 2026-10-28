# Figures and tables of the paper

```bash
cd analysis
uv sync                          # once: creates .venv from pyproject.toml / uv.lock
uv run python make_all.py        # regenerates everything and copies it to paper/figs and paper/tables
```

Generated files go to `analysis/output/` (not versioned). Figures need the Latin Modern font of MiKTeX / TeX Live
(found automatically; override with `LATIN_MODERN_FONT`).

| Script | Produces |
|---|---|
| `configurations.py` | Definition of the 46 configurations (IDs, parameters, sheet names) and of the repository paths |
| `latency_analysis.py` | Figure 12 (`latency_all_configurations`), Figure 13 (`premature_alarm_rate_*`), Figures 14, 15a, 16a (`latency_*_vs_*`), Tables A.1–A.9 (`median_latency_all_*.tex`), `results_aggregated.xlsx` |
| `pairwise_comparison.py` | Figures 15b and 16b (`pairwise_*`), `comparison_results.xlsx` |
| `theory_vs_observed.py` | Ideal-case prediction vs. observed (values quoted in Section 4.5.3) |
| `ideal_latency.py` | Table 9 (`ideal_latency.tex`) and Figure 16c (`ideal_vs_observed_latency`): latency with ideal models (`results/positive_videos/ideal_latency.csv`, made by `experiments/ideal_latency.py`) vs. Eq. (4) vs. observed |
| `false_alarm_tables.py` | Table 8 (`falsealarm_neg_summary.tex`) and Table A.10 (`falsealarm_neg_per_video.tex`) |
| `summary_table.py` | Table 10 (`summary_all_methods.tex`, all configurations and methods from the literature) |
| `method_comparison.py` | Loads the workbooks of every method (used by the two table scripts) and writes `method_comparison.xlsx` |
| `timing_diagram.py` | Figure 3 (`timing_two_step_vs_best_only`) |
| `make_all.py` | Runs the scripts above and copies the paper figures and tables to `paper/` |
| `import_runs.py` | Adds the runs of a new experiment workbook to `results/` |
| `model_complexity.py` | GFLOPs column of Table 6: `export` turns the trained `.pt` checkpoints into ONNX (Ultralytics 8.3.99, as deployed), `count` gives parameters, MACs and FLOPs with `onnx-opcounter` 0.1.0 and checks the exports against `models/saferfire/` (`output/model_complexity.csv`; commands in the script docstring) |
| `alarm_frames/premature_alarm_figures.py` | Figure 17a–f (`paper/figs/premature_alarms/`), needs the videos, the models and `ultralytics` |
| `alarm_frames/false_alarm_figures.py` | Figure 17g–l (`paper/figs/negative_false_alarms/`), needs the videos, the models and `onnxruntime` |

The other figures of the paper (diagrams, dataset samples, LayerCAM maps, video thumbnails, notifications,
training curves) are fixed images in `paper/figs/`.
