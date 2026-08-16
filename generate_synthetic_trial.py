"""Generate synthetic clinical-trial data, validate, and produce plots.

    python generate_synthetic_trial.py [--patients 500] [--seed 20260817]

Outputs:
    synthetic_trial/data/          — CSV tables
    synthetic_trial/reports/       — validation + scenario distribution JSON
    synthetic_trial/plots/         — per-scenario trajectory plots
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# Add synthetic_trial/src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "synthetic_trial" / "src"))

from generator import (                  # noqa: E402
    SCENARIOS, SIGNAL_NAMES, WINDOW_MINUTES,
    SyntheticTrialGenerator,
)
from validate import validate_dataset    # noqa: E402

# ── Output directories ───────────────────────────────────────────────────────

SYNTH_ROOT   = SCRIPT_DIR / "synthetic_trial"
DATA_DIR     = SYNTH_ROOT / "data"
REPORTS_DIR  = SYNTH_ROOT / "reports"
PLOTS_DIR    = SYNTH_ROOT / "plots" / "scenario_trajectories"


def ensure_dirs():
    for d in (DATA_DIR, REPORTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ── Plotting ─────────────────────────────────────────────────────────────────

# Consistent colours for each signal
SIGNAL_COLORS = {
    "heart_rate":       "#E74C3C",
    "spo2":             "#2980B9",
    "respiratory_rate": "#27AE60",
    "systolic_bp":      "#8E44AD",
    "diastolic_bp":     "#D35400",
    "temperature":      "#F39C12",
}

SIGNAL_LABELS = {
    "heart_rate":       "Heart Rate (bpm)",
    "spo2":             "SpO₂ (%)",
    "respiratory_rate": "Respiratory Rate",
}


def plot_scenario(observations_df, scenario, output_path):
    """Plot HR, SpO2, RR for one representative patient of this scenario."""
    subset = observations_df[observations_df["scenario"] == scenario]
    if subset.empty:
        return

    pid = subset["patient_id"].iloc[0]
    patient_obs = subset[subset["patient_id"] == pid].sort_values("timestamp")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Scenario: {scenario}\nPatient {pid}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    signals = ["heart_rate", "spo2", "respiratory_rate"]
    for ax, sig in zip(axes, signals):
        ts = patient_obs["timestamp"].values
        vals = patient_obs[sig].values
        color = SIGNAL_COLORS[sig]

        ax.plot(ts, vals, color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(SIGNAL_LABELS[sig], fontsize=10)
        ax.grid(True, alpha=0.3)

        # Shade low-coverage regions
        low_cov = patient_obs[patient_obs["coverage_percent"] < 50]
        for _, row in low_cov.iterrows():
            ax.axvspan(row["timestamp"], row["timestamp"] + WINDOW_MINUTES,
                       alpha=0.25, color="red", linewidth=0)

        # Mark dose boundaries
        from generator import WINDOWS_PER_DOSE, DOSES_PER_PATIENT
        for d in range(1, DOSES_PER_PATIENT):
            boundary = d * WINDOWS_PER_DOSE * WINDOW_MINUTES
            ax.axvline(boundary, color="gray", linestyle="--",
                       alpha=0.5, linewidth=0.8)
            if sig == signals[0]:
                ax.text(boundary + 2, ax.get_ylim()[1], f"Dose {d + 1}",
                        fontsize=8, color="gray", va="top")

    axes[-1].set_xlabel("Time (minutes from enrollment)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)

    ensure_dirs()

    # ── Generate ─────────────────────────────────────────────────────────────
    rule("GENERATING SYNTHETIC CLINICAL-TRIAL DATA")
    print(f"  patients : {args.patients}")
    print(f"  seed     : {args.seed}")

    gen = SyntheticTrialGenerator(n_patients=args.patients, seed=args.seed)
    tables = gen.generate()

    for name, df in tables.items():
        print(f"  {name:<22} : {df.shape[0]:>7} rows × {df.shape[1]} cols")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    rule("SAVING DATA")
    for name, df in tables.items():
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path}")

    # ── Scenario distribution ────────────────────────────────────────────────
    rule("SCENARIO DISTRIBUTION")
    dist = dict(Counter(tables["patients"]["scenario"]))
    for scenario in SCENARIOS:
        count = dist.get(scenario, 0)
        pct = 100 * count / args.patients
        bar = "█" * int(pct / 2)
        print(f"  {scenario:<28} {count:>4}  ({pct:5.1f}%)  {bar}")

    dist_path = REPORTS_DIR / "scenario_distribution.json"
    dist_path.write_text(json.dumps(dist, indent=2), encoding="utf-8")
    print(f"\n  → {dist_path}")

    # ── Validation ───────────────────────────────────────────────────────────
    rule("VALIDATION")
    results = validate_dataset(tables, seed=args.seed, n_patients=args.patients)

    all_passed = True
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status}  {r['name']:<40} {r['detail']}")
        if not r["passed"]:
            all_passed = False

    report_path = REPORTS_DIR / "validation_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  → {report_path}")

    if all_passed:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — see report for details")

    # ── Plots ────────────────────────────────────────────────────────────────
    rule("GENERATING SCENARIO TRAJECTORY PLOTS")
    for scenario in SCENARIOS:
        fname = scenario.lower() + ".png"
        path = PLOTS_DIR / fname
        plot_scenario(tables["observations"], scenario, path)
        print(f"  {path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    rule("EVENT SUMMARY")
    events = tables["events"]
    if events.empty:
        print("  No events generated.")
    else:
        event_dist = dict(Counter(events["event_type"]))
        for etype, count in sorted(event_dist.items(), key=lambda x: -x[1]):
            print(f"  {etype:<25} : {count}")
        print(f"\n  total events : {len(events)}")

    rule("DATASET SHAPES")
    for name, df in tables.items():
        print(f"  {name:<22} : {df.shape}")

    rule("DONE")
    print("  Synthetic clinical-trial dataset generated successfully.")
    print("  No clinical validity is claimed.")
    print("  All scenario labels and ground-truth states are retained")
    print("  for downstream evaluation.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
