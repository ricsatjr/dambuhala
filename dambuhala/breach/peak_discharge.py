"""
peak_discharge.py
=================
Empirical models for estimating peak outflow from breached embankment dams.

All models in this module require only reservoir characteristics available
prior to breach (Vw and/or Hw), making them suitable for rapid preliminary
assessment without breach geometry data.

Units
-----
    Vw  : reservoir volume above breach invert [m³]
    Hw  : water depth above breach invert [m]
    Qp  : peak outflow [m³/s]
    g   : gravitational acceleration [m/s²], default 9.81

References
----------
Froehlich, D.C. (1995). Peak outflow from breached embankment dam.
    J. Water Resour. Plann. Manage., 121(1), 90–97.

Wang, B., Chen, Y., Wu, C., Peng, Y., Song, J., Liu, W., Liu, X. (2018).
    Empirical and semi-analytical models for predicting peak outflows caused
    by embankment dam failures. J. Hydrology, 562, 692–702.
"""

import math


# ---------------------------------------------------------------------------
# Individual model functions
# ---------------------------------------------------------------------------

def kirkpatrick_1977(Hw: float) -> float:
    """
    Kirkpatrick (1977) — depth only.

    Qp = 1.268 * (Hw + 0.3)^2.5

    R² = 0.61, RRMSE = 0.175 (Wang et al. 2018, 40-case validation).
    Tends to under-predict. Uses Hw only.
    """
    return 1.268 * (Hw + 0.3) ** 2.5


def bureau_of_reclamation_1982(Hw: float) -> float:
    """
    Bureau of Reclamation (1982) — depth only.

    Qp = 19.1 * Hw^1.85

    R² = 0.71, RRMSE = 0.190. Uses Hw only.
    """
    return 19.1 * Hw ** 1.85


def macdonald_langridge_monopolis_1984(Vw: float, Hw: float) -> float:
    """
    MacDonald & Langridge-Monopolis (1984) — dam factor.

    Qp = 1.154 * (Vw * Hw)^0.412

    R² = 0.82, RRMSE = 0.139. Calibrated on mixed embankment and concrete
    dam data; may over-predict for pure earthfill dams.
    """
    return 1.154 * (Vw * Hw) ** 0.412


def costa_1985(Vw: float, Hw: float) -> float:
    """
    Costa (1985) — dam factor.

    Qp = 0.981 * (Vw * Hw)^0.42

    R² = 0.83, RRMSE = 0.135. Also calibrated on mixed dam types;
    similar caution as MacDonald & Langridge-Monopolis applies.
    """
    return 0.981 * (Vw * Hw) ** 0.42


def evans_1986(Vw: float) -> float:
    """
    Evans (1986) — volume only.

    Qp = 0.72 * Vw^0.53

    R² = 0.70, RRMSE = 0.186. Uses Vw only; lower accuracy than
    models that also incorporate Hw.
    """
    return 0.72 * Vw ** 0.53


def froehlich_1995(Vw: float, Hw: float) -> float:
    """
    Froehlich (1995) — regression on 22 embankment dam failures.

    Qp = 0.607 * Vw^0.295 * Hw^1.24

    R² = 0.92, RRMSE = 0.074. Uncertainty band ≈ ±1/3 order of magnitude
    (±2 Se = ±0.425). Failure mode and embankment width were found not to
    improve the regression significantly (Froehlich 1995).
    """
    return 0.607 * Vw ** 0.295 * Hw ** 1.24


def webby_1996(Vw: float, Hw: float, g: float = 9.81) -> float:
    """
    Webby (1996) — dimensionless formulation.

    Derived from the same dimensionless framework as Wang et al. (2018):

        Qp / sqrt(g * Vw^(5/3)) = f(Hw / Vw^(1/3))

    which reduces to:

        Qp = 0.0443 * sqrt(g) * Vw^0.365 * Hw^1.405

    R² = 0.95, RRMSE = 0.059. High accuracy; comparable to Wang et al.
    """
    return 0.0443 * math.sqrt(g) * Vw ** 0.365 * Hw ** 1.405


def wang_2018(Vw: float, Hw: float, g: float = 9.81) -> float:
    """
    Wang et al. (2018) Equation 4 — dimensionless regression on 40 failures.

    Derived from:

        Qp / sqrt(g * Vw^(5/3)) = 0.0370 * (Vw / Hw^3)^(-0.4262)

    which gives:

        Qp = 0.0370 * sqrt(g * Vw^(5/3)) * (Vw / Hw^3)^(-0.4262)

    R² = 0.96, RRMSE = 0.056, em = 0.000, ±2Se = ±0.319.
    Best-performing purely empirical model using only Vw and Hw.
    """
    return 0.0370 * math.sqrt(g * Vw ** (5 / 3)) * (Vw / Hw ** 3) ** (-0.4262)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

#: Model registry — ordered from weakest to strongest performance (R²).
#: Each entry: (function, required_args)
_MODELS = {
    "Kirkpatrick (1977)":              (kirkpatrick_1977,                ["Hw"]),
    "Bureau of Reclamation (1982)":    (bureau_of_reclamation_1982,      ["Hw"]),
    "Evans (1986)":                    (evans_1986,                      ["Vw"]),
    "MacDonald & L-M (1984)":         (macdonald_langridge_monopolis_1984, ["Vw", "Hw"]),
    "Costa (1985)":                    (costa_1985,                      ["Vw", "Hw"]),
    "Froehlich (1995)":                (froehlich_1995,                  ["Vw", "Hw"]),
    "Webby (1996)":                    (webby_1996,                      ["Vw", "Hw"]),
    "Wang et al. (2018)":              (wang_2018,                       ["Vw", "Hw"]),
}

#: Published performance statistics on Wang et al. (2018) 40-case dataset.
MODEL_STATS = {
    "Kirkpatrick (1977)":           {"R2": 0.6088, "RRMSE": 0.1752, "em": -0.2677, "Se": 0.4368},
    "Bureau of Reclamation (1982)": {"R2": 0.7053, "RRMSE": 0.1900, "em":  0.1375, "Se": 0.4228},
    "Evans (1986)":                 {"R2": 0.6970, "RRMSE": 0.1861, "em":  0.1606, "Se": 0.4213},
    "MacDonald & L-M (1984)":      {"R2": 0.8217, "RRMSE": 0.1393, "em":  0.0434, "Se": 0.3431},
    "Costa (1985)":                 {"R2": 0.8295, "RRMSE": 0.1347, "em":  0.0357, "Se": 0.3364},
    "Froehlich (1995)":             {"R2": 0.9238, "RRMSE": 0.0740, "em": -0.0770, "Se": 0.2125},
    "Webby (1996)":                 {"R2": 0.9537, "RRMSE": 0.0593, "em": -0.0608, "Se": 0.1655},
    "Wang et al. (2018)":           {"R2": 0.9620, "RRMSE": 0.0555, "em":  0.0000, "Se": 0.1597},
}


def peak_discharge(
    Vw: float,
    Hw: float,
    g: float = 9.81,
    models: list | None = None,
) -> dict:
    """
    Estimate peak breach outflow using multiple empirical models.

    Parameters
    ----------
    Vw : float
        Volume of water stored above the breach invert at the time of failure
        [m³]. Note: 1 million m³ = 1 × 10⁶ m³.
    Hw : float
        Depth of water above the breach invert at the time of failure [m].
    g : float, optional
        Gravitational acceleration [m/s²]. Default is 9.81.
    models : list of str, optional
        Subset of model names to run. Defaults to all registered models.
        See ``peak_discharge.list_models()`` for valid names.

    Returns
    -------
    dict
        ``{model_name: Qp_m3s}`` — peak outflow in m³/s for each model.

    Raises
    ------
    ValueError
        If Vw or Hw are not positive.

    Examples
    --------
    >>> from dambuhala.breach.peak_discharge import peak_discharge
    >>> results = peak_discharge(Vw=22.8e6, Hw=28.0)
    >>> for name, qp in results.items():
    ...     print(f"{name:<30s}  {qp:>10.1f} m³/s")
    """
    if Vw <= 0 or Hw <= 0:
        raise ValueError(f"Vw and Hw must be positive; got Vw={Vw}, Hw={Hw}.")

    target = models if models is not None else list(_MODELS.keys())
    results = {}

    for name in target:
        if name not in _MODELS:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Valid options: {list(_MODELS.keys())}"
            )
        fn, args = _MODELS[name]
        kwargs = {"g": g} if "g" in fn.__code__.co_varnames else {}
        call_args = []
        if "Vw" in args:
            call_args.append(Vw)
        if "Hw" in args:
            call_args.append(Hw)
        results[name] = fn(*call_args, **kwargs)

    return results


def peak_discharge_range(Vw: float, Hw: float, g: float = 9.81) -> dict:
    """
    Return the min, median, and max predicted peak discharge across all models,
    together with the full model results.

    Useful for communicating the spread of empirical uncertainty without
    selecting a single model.

    Parameters
    ----------
    Vw : float
        Volume of water above breach invert [m³].
    Hw : float
        Depth of water above breach invert [m].
    g : float, optional
        Gravitational acceleration [m/s²].

    Returns
    -------
    dict with keys:
        ``models``  — full ``{model_name: Qp}`` dict
        ``min``     — minimum Qp across models [m³/s]
        ``median``  — median Qp across models [m³/s]
        ``max``     — maximum Qp across models [m³/s]
    """
    results = peak_discharge(Vw=Vw, Hw=Hw, g=g)
    values = sorted(results.values())
    n = len(values)
    mid = n // 2
    median = (values[mid - 1] + values[mid]) / 2 if n % 2 == 0 else values[mid]
    return {
        "models": results,
        "min":    values[0],
        "median": median,
        "max":    values[-1],
    }


def list_models() -> list:
    """Return the list of registered model names."""
    return list(_MODELS.keys())
