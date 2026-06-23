"""
run_peak_discharge.py
=====================
CLI for estimating peak breach outflow using multiple empirical models.

Runs all registered models (or a user-specified subset) for given reservoir
conditions and prints a ranked results table.  Optionally saves results to CSV.

Usage
-----
# All models
python run_peak_discharge.py --Vw 22.8e6 --Hw 28.0

# Subset of models
python run_peak_discharge.py --Vw 22.8e6 --Hw 28.0 \\
    --models "Froehlich (1995)" "Wang et al. (2018)"

# Save to CSV
python run_peak_discharge.py --Vw 22.8e6 --Hw 28.0 --out outputs/

# List available model names
python run_peak_discharge.py --list-models

Units: Vw in m³, Hw in m, Qp in m³/s.
"""

import argparse
import os
import sys

try:
    from breach.peak_discharge import (
        peak_discharge,
        peak_discharge_range,
        list_models,
        MODEL_STATS,
    )
except ModuleNotFoundError:
    from peak_discharge import (
        peak_discharge,
        peak_discharge_range,
        list_models,
        MODEL_STATS,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

COL_NAME  = 30
COL_QP    = 14
COL_R2    = 8
COL_RRMSE = 8

DIVIDER = "-" * (COL_NAME + COL_QP + COL_R2 + COL_RRMSE + 6)


def print_table(results: dict):
    """Print model results ranked by Qp, with performance statistics."""
    header = (
        f"{'Model':<{COL_NAME}}"
        f"{'Qp (m³/s)':>{COL_QP}}"
        f"{'R²':>{COL_R2}}"
        f"{'RRMSE':>{COL_RRMSE}}"
    )
    print()
    print(header)
    print(DIVIDER)

    for name, qp in sorted(results.items(), key=lambda x: x[1]):
        stats = MODEL_STATS.get(name, {})
        r2    = f"{stats['R2']:.4f}"    if stats else "—"
        rrmse = f"{stats['RRMSE']:.4f}" if stats else "—"
        print(
            f"{name:<{COL_NAME}}"
            f"{qp:>{COL_QP},.1f}"
            f"{r2:>{COL_R2}}"
            f"{rrmse:>{COL_RRMSE}}"
        )

    print(DIVIDER)


def print_envelope(envelope: dict):
    print(f"  Min    : {envelope['min']:>12,.1f} m³/s")
    print(f"  Median : {envelope['median']:>12,.1f} m³/s")
    print(f"  Max    : {envelope['max']:>12,.1f} m³/s")
    print()


def save_csv(results: dict, envelope: dict, Vw: float, Hw: float, path: str):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Inputs"])
        writer.writerow(["Vw_m3", Vw])
        writer.writerow(["Hw_m",  Hw])
        writer.writerow([])
        writer.writerow(["model", "Qp_m3s", "R2", "RRMSE"])
        for name, qp in sorted(results.items(), key=lambda x: x[1]):
            stats = MODEL_STATS.get(name, {})
            r2    = stats.get("R2",    "")
            rrmse = stats.get("RRMSE", "")
            writer.writerow([name, f"{qp:.4f}", r2, rrmse])
        writer.writerow([])
        writer.writerow(["envelope_min_m3s",    f"{envelope['min']:.4f}"])
        writer.writerow(["envelope_median_m3s", f"{envelope['median']:.4f}"])
        writer.writerow(["envelope_max_m3s",    f"{envelope['max']:.4f}"])
    print(f"Results saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate peak breach outflow using empirical models."
    )
    parser.add_argument("--Vw", type=float,
                        help="Reservoir volume above breach invert (m³)")
    parser.add_argument("--Hw", type=float,
                        help="Water depth above breach invert (m)")
    parser.add_argument("--g",  type=float, default=9.81,
                        help="Gravitational acceleration (m/s², default: 9.81)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to run (default: all). "
                             "Use --list-models to see valid names.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for CSV (optional)")
    parser.add_argument("--list-models", action="store_true",
                        help="Print available model names and exit")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name in list_models():
            stats = MODEL_STATS.get(name, {})
            r2    = f"R²={stats['R2']:.4f}" if stats else ""
            print(f"  {name:<30s}  {r2}")
        print()
        sys.exit(0)

    if args.Vw is None or args.Hw is None:
        parser.error("--Vw and --Hw are required (or use --list-models).")

    print(f"\nInputs")
    print(f"  Vw = {args.Vw:,.0f} m³")
    print(f"  Hw = {args.Hw} m")
    print(f"  g  = {args.g} m/s²")
    if args.models:
        print(f"  Models: {args.models}")

    results  = peak_discharge(Vw=args.Vw, Hw=args.Hw, g=args.g, models=args.models)
    envelope = peak_discharge_range(Vw=args.Vw, Hw=args.Hw, g=args.g)

    print_table(results)
    print("Uncertainty envelope (all models):")
    print_envelope(envelope)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        csv_path = os.path.join(args.out, "peak_discharge.csv")
        save_csv(results, envelope, args.Vw, args.Hw, csv_path)


if __name__ == "__main__":
    main()
