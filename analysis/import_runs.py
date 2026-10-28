"""
Append the runs of a workbook written by an experiment script to the results workbook
of its method, so that results/ always holds one workbook per method and video set.

    python import_runs.py <results.xlsx from the RPi5> <results workbook> [--legacy GROUP] [--dry-run]

Examples
    # SAFERFIRE, positive videos, written by experiments/saferfire_positive.py
    python import_runs.py run/results.xlsx ../results/positive_videos/saferfire_latency.xlsx
    # Venancio et al., negative videos (new workbook if it does not exist yet)
    python import_runs.py run/results.xlsx ../results/negative_videos/venancio_false_alarms.xlsx
    # runs of the old launch scripts (truncated sheet names, see LEGACY below)
    python import_runs.py "results (3).xlsx" ../results/positive_videos/saferfire_latency.xlsx --legacy extra

Every sheet of the new workbook is matched by name with a sheet of the results workbook
(sheet names in results/README.md); its rows (one per run) are appended and the "run"
column is renumbered. The video columns must be the same. The "details" sheet of the
negative-video workbooks is appended as well and the "summary" sheet is recomputed.
Sheets without a match are added as new configurations (listed in the output).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SKIP = {"summary"}


def fc(c):
    c = float(c)
    return "0.125" if abs(c - 0.125) < 1e-9 or abs(c - 0.12) < 1e-9 else f"{c:.2f}"


# ------------------------------------------------------------------
# Old sheet names (launch scripts before October 2026). The SAFERFIRE positive-video names were
# truncated to 31 characters by Excel, so "..._Hconf0", "..._Hcon_1", ... depend on the set of
# configurations of the run: "main" = thresholds 0.25/0.50, "extra" = thresholds 0.125/0.75.
# ------------------------------------------------------------------
TS_SUFFIX = {
    "main": {("0.25", "Hconf0"): "0.25", ("0.25", "Hcon_1"): "0.50", ("0.50", "Hconf0"): "0.50"},
    "extra": {("0.125", "Hconf0"): "0.125", ("0.125", "Hcon_1"): "0.25", ("0.125", "Hcon_2"): "0.50",
              ("0.125", "Hcon_3"): "0.75", ("0.25", "Hconf0"): "0.75", ("0.50", "Hconf0"): "0.75",
              ("0.75", "Hconf0"): "0.75"},
}


def legacy_name(sheet, group):
    if m := re.fullmatch(r"light_c(?:onf)?([\d.]+)_N(\d)_M(\d)", sheet):
        return f"LO_c{fc(m[1])}_W{m[2]}T{m[3]}"
    if m := re.fullmatch(r"heavy_c(?:onf)?([\d.]+)", sheet):
        return f"BO_c{fc(m[1])}"
    if m := re.fullmatch(r"LH_L([\d.]+)_N(\d)_M(\d)_H([\d.]+)", sheet):           # negative videos
        return f"TS_cL{fc(m[1])}_cH{fc(m[4])}_W{m[2]}T{m[3]}"
    if m := re.fullmatch(r"two_step_Lconf([\d.]+)_N(\d)_M(\d)_(H\w+)", sheet):     # positive videos
        if group is None:
            sys.exit(f"sheet {sheet!r} has a truncated old name: say which run it is with --legacy main/extra")
        c_l = fc(m[1])
        c_h = TS_SUFFIX[group][(c_l, m[4])]
        return f"TS_cL{c_l}_cH{c_h}_W{m[2]}T{m[3]}"
    if m := re.fullmatch(r"venancio_conf([\d.]+)_area([\d.]+)_win(\d+)", sheet):
        win = int(m[3]) * 10 if int(m[3]) < 10 else int(m[3])
        return f"AVT_c{float(m[1]):.2f}_area{float(m[2]):.2f}_win{win}"
    if m := re.fullmatch(r"onfire_ff_thr([\d.]+)_N1_M1", sheet):
        return f"FF_thr{float(m[1]):.2f}"
    if m := re.fullmatch(r"onfire_sp_thr([\d.]+)_k(\d+)", sheet):
        return f"SP_thr{float(m[1]):.2f}_k{m[2]}"
    if m := re.match(r"(edge|off)_fl_c([\d.]+)_fg([\d.]+)_tC([\d.]+)", sheet):   # FLAME (tO = 0.1 s)
        regime = "edge" if m[1] == "edge" else "offline"
        return f"{regime}_c{float(m[2]):.2f}_fg{float(m[3]):.2f}_tC{float(m[4]):.1f}"
    if m := re.fullmatch(r"paper_(raw|size|size_hsv)", sheet):                    # Dumitru & Spinu
        return f"{m[1]}_K1"
    if m := re.fullmatch(r"paper_(size|full)_k(\d)", sheet):
        return f"{m[1]}_K{m[2]}"
    return sheet


def summary(sheets):
    rows = []
    for name, df in sheets.items():
        v = df.drop(columns="run").to_numpy(float).ravel()
        n, k = int((v == v).sum()), int((v == 1).sum())
        rows.append({"experiment": name, "runs": n, "false_alarms": k, "no_alarm": n - k,
                     "false_alarm_rate": k / n if n else float("nan")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", type=Path, help="workbook written by the experiment script")
    ap.add_argument("target", type=Path, help="results workbook of the method (created if missing)")
    ap.add_argument("--legacy", choices=["main", "extra"],
                    help="translate old sheet names (main: thresholds 0.25/0.50, extra: 0.125/0.75)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be added without writing")
    args = ap.parse_args()

    old = {}
    if args.target.exists():
        x = pd.ExcelFile(args.target)
        old = {s: pd.read_excel(x, s) for s in x.sheet_names}

    x = pd.ExcelFile(args.new)
    report = []
    for s in x.sheet_names:
        if s in SKIP or s.startswith("partial"):
            continue
        df = pd.read_excel(x, s)
        name = legacy_name(s, args.legacy)
        if s == "details":
            if "experiment" in df:
                df["experiment"] = df["experiment"].map(lambda e: legacy_name(e, args.legacy))
            old["details"] = pd.concat([old.get("details", pd.DataFrame()), df], ignore_index=True)
            report.append(f"details: +{len(df)} rows")
            continue
        if "run" not in df:
            sys.exit(f"sheet {s!r} has no 'run' column")
        if name in old:
            prev = old[name]
            videos_prev, videos_new = set(prev.columns) - {"run"}, set(df.columns) - {"run"}
            if videos_prev != videos_new:
                sys.exit(f"sheet {s!r} -> {name!r}: different videos "
                         f"(missing {sorted(videos_prev - videos_new)}, extra {sorted(videos_new - videos_prev)})")
            df = df[prev.columns]
            merged = pd.concat([prev.drop(columns="run"), df.drop(columns="run")], ignore_index=True)
            report.append(f"{name}: {len(prev)} + {len(df)} runs")
        else:
            merged = df.drop(columns="run").reset_index(drop=True)
            report.append(f"{name}: NEW configuration, {len(df)} runs")
        merged.insert(0, "run", range(1, len(merged) + 1))
        old[name] = merged

    print("\n".join(report))
    if args.dry_run:
        print("(dry run: nothing written)")
        return

    configs = {k: v for k, v in old.items() if k not in ("summary", "details")}
    with pd.ExcelWriter(args.target, engine="openpyxl") as w:
        for name, df in configs.items():
            df.to_excel(w, sheet_name=name, index=False)
        if "details" in old:
            summary(configs).to_excel(w, sheet_name="summary", index=False)
            old["details"].to_excel(w, sheet_name="details", index=False)
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
