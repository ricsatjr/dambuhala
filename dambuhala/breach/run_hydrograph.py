"""
run_hydrograph.py
=================
CLI for generating a dam breach hydrograph using hydrograph_BetaDist.

Filters candidate Beta distribution shapes to those whose peak discharge
falls within a user-specified normalised time window [t1, t2], where
t_norm=0 is the start and t_norm=1 is the end of the hydrograph.

Outputs
-------
<output_dir>/hydrograph.csv   -- time (seconds) and discharge (m3/s) columns
<output_dir>/hydrograph.png   -- t vs Q plot

Usage
-----
python run_hydrograph.py --Vw 50e6 --Qp 5000 --dt 0.1 --t1 0.2 --t2 0.4

All units SI: Vw in m3, Qp in m3/s, dt in hours, CSV time output in seconds.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Resolve import -- works whether called from repo root or breach/ directory
# ---------------------------------------------------------------------------
try:
    from breach.hydrograph import hydrograph_BetaDist
except ModuleNotFoundError:
    from hydrograph import hydrograph_BetaDist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_by_peak_location(h, t1, t2):
    """Return (a, b) pairs whose beta PDF peak falls in [t1, t2]."""
    filtered = []
    for a, b in h.valid_beta_params:
        x, pdf = h.make_points(a, b)
        peak_loc = x[np.argmax(pdf)]
        if t1 <= peak_loc <= t2:
            filtered.append((a, b))
    return filtered


def try_make_hydrograph(h, a, b, initial_guess):
    """
    Attempt make_hydrograph for a given (a, b) pair.

    Pre-calls optimize_duration with a physics-based initial_guess to steer
    the optimizer, then calls make_hydrograph.  Returns (t, q) on success,
    None on failure.
    """
    try:
        # Pre-optimise with a sensible starting point.
        # make_hydrograph calls optimize_duration internally with its own
        # default (24 h), so we call it first to seed h.T -- but since
        # make_hydrograph overwrites h.T we instead check T here and skip
        # pairs that converge to a degenerate duration.
        T = h.optimize_duration(a, b, initial_guess=initial_guess)

        min_T = 3 * h.dt   # need at least a few timesteps for the spline
        if T < min_T:
            print(f"  skipping a={a}, b={b}: optimised T={T:.4f} h is too short")
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t, q, _ = h.make_hydrograph(a, b, plot=False)

        # Sanity check: discharge array must be finite and non-degenerate
        if len(t) < 3 or not np.all(np.isfinite(q)) or np.max(q) == 0:
            print(f"  skipping a={a}, b={b}: degenerate discharge array")
            return None

        # Clip spline artefacts -- UnivariateSpline can produce small negative
        # values at the tails where discharge approaches zero.
        n_negative = int(np.sum(q < 0))
        if n_negative > 0:
            print(f"  warning: {n_negative} negative discharge value(s) clipped "
                  f"to zero (spline artefact)")
        q = np.maximum(q, 0.0)

        return t, q

    except Exception as exc:
        print(f"  skipping a={a}, b={b}: {exc}")
        return None


def volume_check(t, q, Vw):
    """
    Integrate q over t to compute the hydrograph volume and compare with Vw.

    t is in hours, q in m3/s -- multiply t by 3600 to get seconds before
    integrating so the result is in m3.
    """
    t_sec = t * 3600.0
    V_computed = np.trapz(q, t_sec)
    error_abs = V_computed - Vw
    error_pct = 100.0 * error_abs / Vw
    print(f"\nVolume check")
    print(f"  Target   : {Vw:>15,.0f} m3")
    print(f"  Computed : {V_computed:>15,.0f} m3")
    print(f"  Error    : {error_abs:>+15,.0f} m3  ({error_pct:+.2f}%)")
    return V_computed, error_pct


def save_csv(t, q, path):
    """Save hydrograph to CSV with time in seconds."""
    header = "time_s,discharge_m3s"
    data = np.column_stack([t * 3600.0, q])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"CSV saved: {path}")


def save_plot(t, q, a, b, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t * 3600.0, q, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(f"Breach hydrograph  --  Beta: a={a:.2f}, b={b:.2f}")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved:  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a dam breach hydrograph (Beta distribution shape)."
    )
    parser.add_argument("--Vw",  type=float, required=True,
                        help="Total breach outflow volume (m3)")
    parser.add_argument("--Qp",  type=float, required=True,
                        help="Peak discharge (m3/s)")
    parser.add_argument("--dt",  type=float, default=0.1,
                        help="Timestep in hours (default: 0.1)")
    parser.add_argument("--t1",  type=float, default=0.1,
                        help="Min normalised peak time, 0-1 (default: 0.1)")
    parser.add_argument("--t2",  type=float, default=0.5,
                        help="Max normalised peak time, 0-1 (default: 0.5)")
    parser.add_argument("--n",   type=int,   default=200,
                        help="Beta parameter candidates to sample (default: 200)")
    parser.add_argument("--seed", type=int,  default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", type=str,   default="outputs",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    if not (0 <= args.t1 < args.t2 <= 1):
        sys.exit("Error: t1 and t2 must satisfy 0 <= t1 < t2 <= 1")

    os.makedirs(args.out, exist_ok=True)

    # Physics-based initial guess for duration (triangular approximation)
    initial_guess = 2 * args.Vw / (args.Qp * 3600)
    print(f"Physics-based duration guess: {initial_guess:.2f} h")

    # Build candidate pool
    print(f"Sampling {args.n} Beta parameter pairs ...")
    h = hydrograph_BetaDist(Vw=args.Vw, Qp=args.Qp, dt=args.dt,
                             minpeakloc=args.t1, maxpeakloc=args.t2)
    h.generate_beta_params(n=args.n, randomseed=args.seed)

    # Filter by normalised peak location
    filtered = filter_by_peak_location(h, args.t1, args.t2)
    print(f"Pairs passing peak-location filter [{args.t1}, {args.t2}]: {len(filtered)}")

    if not filtered:
        sys.exit(
            "No valid pairs found. Try widening t1/t2, increasing --n, "
            "or adjusting --t1/--t2."
        )

    # Try each filtered pair until one succeeds
    result = None
    chosen_a, chosen_b = None, None
    for a, b in filtered:
        print(f"Trying a={a}, b={b} ...")
        result = try_make_hydrograph(h, a, b, initial_guess)
        if result is not None:
            chosen_a, chosen_b = a, b
            break

    if result is None:
        sys.exit(
            "All filtered pairs failed. Try increasing --n or widening the "
            "peak location window."
        )

    t, q = result
    print(f"Hydrograph generated: a={chosen_a}, b={chosen_b}, "
          f"T={h.T:.2f} h, Qp={np.max(q):.1f} m3/s")

    volume_check(t, q, args.Vw)

    print()
    save_csv(t, q, os.path.join(args.out, "hydrograph.csv"))
    save_plot(t, q, chosen_a, chosen_b, os.path.join(args.out, "hydrograph.png"))


if __name__ == "__main__":
    main()
