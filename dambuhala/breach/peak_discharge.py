"""
breach/peak_discharge.py
========================
Empirical peak discharge models for dam breach flood estimation.

All models estimate peak outflow discharge (Qp, m³/s) from two primary inputs:
    Vw  — reservoir volume above breach invert at time of failure (m³)
    Hw  — water height above breach invert at time of failure (m)

Individual model functions follow a consistent signature:
    peak_discharge_<author>(Vw, Hw, ...) -> float

The unified function ``peak_discharge_all`` runs every implemented model and
returns per-model results together with a min / median / max uncertainty
envelope.

References
----------
Wang et al. (2018)
    Wang, B., Chen, Y., Wu, C., Peng, Y., Ma, X., Song, J. (2018). Empirical
    and semi-analytical models for predicting peak outflows caused by embankment
    dam failures. Journal of Hydrology, 562, 692–702.
    https://doi.org/10.1016/j.jhydrol.2018.05.049  [Eq. 4]

Froehlich (1995)
    Froehlich, D.C. (1995). Peak outflow from breached embankment dam. Journal
    of Water Resources Planning and Management, 121(1), 90–97.

MacDonald & Langridge-Monopolis (1984)
    MacDonald, T.C., Langridge-Monopolis, J. (1984). Breaching characteristics
    of dam failures. Journal of Hydraulic Engineering, 110(5), 567–586.

Von Thun & Gillette (1990)
    Von Thun, J.L., Gillette, D.R. (1990). Guidance on breach parameters.
    Internal memorandum, U.S. Bureau of Reclamation, Denver, Colorado.

Pierce et al. (2010)
    Pierce, M.W., Thornton, C.I., Abt, S.R. (2010). Predicting peak outflow
    from breached embankment dams. Journal of Hydrologic Engineering, 15(5),
    338–349.
"""

from __future__ import annotations

import statistics
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_G = 9.81          # gravitational acceleration (m/s²)
_M3_PER_MILLION = 1e6  # conversion: 1 million m³ = 1e6 m³

# Threshold used in the Von Thun & Gillette (1990) regime selector (m)
_VTG_THRESHOLD = 1.24e6  # Vw / Hw² < this → small-reservoir regime (m)


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------

def peak_discharge_wang(Vw: float, Hw: float, g: float = _G) -> float:
    """
    Peak breach discharge — Wang et al. (2018), Eq. 4.

    Semi-analytical model derived from energy and momentum principles,
    validated against a global dam-failure database.

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).
    g  : float, optional
        Gravitational acceleration (m/s²). Default 9.81.

    Returns
    -------
    float
        Peak outflow discharge Qp (m³/s).

    Notes
    -----
    The formula operates on volume in *million* m³ (Vw_M).  The bug present
    in the original dambuhala breach.py — ``*(g*(Vw_M**(5/3)))*0.5`` instead
    of ``*(g*(Vw_M**(5/3)))**0.5`` — has been corrected here.
    """
    if Vw <= 0:
        raise ValueError(f"Vw must be positive, got {Vw}")
    if Hw <= 0:
        raise ValueError(f"Hw must be positive, got {Hw}")

    Vw_M = Vw / _M3_PER_MILLION  # convert m³ → million m³

    term1 = (Vw_M / Hw ** 3) ** (-0.4262)
    term2 = (g * Vw_M ** (5 / 3)) ** 0.5   # ← **0.5, not *0.5

    return 0.0370 * term1 * term2


def peak_discharge_froehlich(Vw: float, Hw: float) -> float:
    """
    Peak breach discharge — Froehlich (1995).

    One of the most widely cited empirical models, derived from 22 embankment
    dam failures.

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).

    Returns
    -------
    float
        Peak outflow discharge Qp (m³/s).
    """
    if Vw <= 0:
        raise ValueError(f"Vw must be positive, got {Vw}")
    if Hw <= 0:
        raise ValueError(f"Hw must be positive, got {Hw}")

    return 0.272 * (Vw ** 0.454) * (Hw ** 1.24)


def peak_discharge_mcdonald(Vw: float, Hw: float) -> float:
    """
    Peak breach discharge — MacDonald & Langridge-Monopolis (1984).

    One of the earliest widely adopted empirical models.  Tends toward
    conservative (higher) estimates and is often used as an upper bound.

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).

    Returns
    -------
    float
        Peak outflow discharge Qp (m³/s).
    """
    if Vw <= 0:
        raise ValueError(f"Vw must be positive, got {Vw}")
    if Hw <= 0:
        raise ValueError(f"Hw must be positive, got {Hw}")

    return 1.154 * (Vw * Hw) ** 0.412


def peak_discharge_von_thun(Vw: float, Hw: float) -> float:
    """
    Peak breach discharge — Von Thun & Gillette (1990).

    Two-regime model: the applicable equation is selected automatically based
    on the dimensionless reservoir index Vw / Hw².

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).

    Returns
    -------
    float
        Peak outflow discharge Qp (m³/s).

    Notes
    -----
    Regime selector:
        Vw / Hw²  <  1.24 × 10⁶ m  →  Qp = 19.1 × Hw^1.85
        Vw / Hw²  ≥  1.24 × 10⁶ m  →  Qp = 1.205 × Vw^0.48
    """
    if Vw <= 0:
        raise ValueError(f"Vw must be positive, got {Vw}")
    if Hw <= 0:
        raise ValueError(f"Hw must be positive, got {Hw}")

    index = Vw / Hw ** 2

    if index < _VTG_THRESHOLD:
        return 19.1 * Hw ** 1.85
    else:
        return 1.205 * Vw ** 0.48


def peak_discharge_pierce(Vw: float, Hw: float) -> float:
    """
    Peak breach discharge — Pierce et al. (2010).

    More recent regression using an expanded dam-failure database.  The
    published equation contains no Hw exponent term (Hw^0.0 = 1), so peak
    discharge depends on reservoir volume only.

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).  Accepted for API consistency
        but has no effect on the result per the published equation.

    Returns
    -------
    float
        Peak outflow discharge Qp (m³/s).
    """
    if Vw <= 0:
        raise ValueError(f"Vw must be positive, got {Vw}")
    if Hw <= 0:
        raise ValueError(f"Hw must be positive, got {Hw}")

    # Hw^0.0 = 1 per Pierce et al. (2010); Hw accepted for signature consistency
    return 0.194 * Vw ** 0.718


# ---------------------------------------------------------------------------
# Unified result container
# ---------------------------------------------------------------------------

class PeakDischargeResult(NamedTuple):
    """
    Container returned by ``peak_discharge_all``.

    Attributes
    ----------
    wang        : float  — Wang et al. (2018) (m³/s)
    froehlich   : float  — Froehlich (1995) (m³/s)
    mcdonald    : float  — MacDonald & Langridge-Monopolis (1984) (m³/s)
    von_thun    : float  — Von Thun & Gillette (1990) (m³/s)
    pierce      : float  — Pierce et al. (2010) (m³/s)
    Qp_min      : float  — minimum across all models (m³/s)
    Qp_median   : float  — median across all models (m³/s)
    Qp_max      : float  — maximum across all models (m³/s)
    """
    wang: float
    froehlich: float
    mcdonald: float
    von_thun: float
    pierce: float
    Qp_min: float
    Qp_median: float
    Qp_max: float


# ---------------------------------------------------------------------------
# Unified function
# ---------------------------------------------------------------------------

def peak_discharge_all(Vw: float, Hw: float, g: float = _G) -> PeakDischargeResult:
    """
    Run all implemented empirical peak discharge models and return an
    uncertainty envelope.

    Parameters
    ----------
    Vw : float
        Reservoir volume above breach invert (m³).
    Hw : float
        Water height above breach invert (m).
    g  : float, optional
        Gravitational acceleration (m/s²). Default 9.81. Passed to Wang et al.
        only; other models do not use g.

    Returns
    -------
    PeakDischargeResult
        Named tuple with individual model outputs and min / median / max
        envelope values (all in m³/s).

    Examples
    --------
    >>> result = peak_discharge_all(Vw=1.2e8, Hw=45.0)
    >>> print(f"Envelope: {result.Qp_min:.0f} – {result.Qp_max:.0f} m³/s")
    >>> print(f"Median:   {result.Qp_median:.0f} m³/s")
    """
    wang      = peak_discharge_wang(Vw, Hw, g=g)
    froehlich = peak_discharge_froehlich(Vw, Hw)
    mcdonald  = peak_discharge_mcdonald(Vw, Hw)
    von_thun  = peak_discharge_von_thun(Vw, Hw)
    pierce    = peak_discharge_pierce(Vw, Hw)

    values = [wang, froehlich, mcdonald, von_thun, pierce]

    return PeakDischargeResult(
        wang=wang,
        froehlich=froehlich,
        mcdonald=mcdonald,
        von_thun=von_thun,
        pierce=pierce,
        Qp_min=min(values),
        Qp_median=statistics.median(values),
        Qp_max=max(values),
    )
