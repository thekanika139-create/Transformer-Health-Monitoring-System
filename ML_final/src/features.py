"""Causal (past-only) feature engineering for THMS.

`build_features(df, window)` turns ONE run of time-ordered sensor rows into the
model feature matrix. Training uses it through `build_features_by_run`; the
live dashboard calls it directly on its rolling buffer. There is exactly one
copy of the feature code.

Causality
---------
Row t of the output depends only on input rows t-window+1 .. t (the current
sample plus window-1 past samples). Every window is trailing: pandas rolling
with center=False, and an explicit trailing view for the slopes. Nothing is
shifted backwards, interpolated or filled from later rows. Consequently, for a
live buffer holding at least `window` samples,
`build_features(buffer).iloc[-1]` equals the row training saw for that sample.

Missing history (first window-1 samples of a run, or a live buffer shorter than window)
---------------------------------------------------------------------------------------
- Features that need only the current sample (raw inputs, load_pu, oil_rise,
  residual, vib_ratio, distances) are always defined.
- vib_roll_mean and h2_roll_max use an expanding window (min_periods=1): over
  the samples seen so far, so row 0 is just the first sample.
- h2_rate_ppm_per_day and oil_temp_slope_C_per_h are NaN until a full window is
  available. A slope from 2-3 points of noisy 15-min data would be misleading,
  and LightGBM handles NaN natively. A NaN anywhere inside the window also
  gives NaN.
- `warmup_mask` flags exactly the rows where a trend feature is NaN so that
  train/evaluate can drop or report them explicitly.

Time axis
---------
Slopes assume regular sampling at config.SAMPLE_PERIOD_H and use the sample
index, so t_hours / hour_of_day are never used as features. `run_id` and
`t_hours` are read only for sanity checks (single run, strictly increasing time).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config as cfg
from src.data import assert_no_forbidden_columns

__all__ = [
    "RAW_FEATURES",
    "ENGINEERED_FEATURES",
    "FEATURE_NAMES",
    "MONOTONE_DECREASING",
    "get_feature_names",
    "get_monotone_constraints",
    "FEATURE_WEIGHT_GROUPS",
    "get_feature_contri",
    "build_features",
    "build_features_by_run",
    "warmup_mask",
]

# --------------------------------------------------------------------------
# Feature registry
# --------------------------------------------------------------------------
RAW_FEATURES: tuple[str, ...] = tuple(cfg.INPUT_FEATURES)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "load_pu",
    "oil_rise",
    "in_out_resid",
    "abs_in_out_resid",
    "vib_ratio",
    "d_h2",
    "d_oil_temp",
    "d_load",
    "d_vib",
    "d_abs_resid",
    "volt_dev",
    "d_volt",
    "h2_rate_ppm_per_day",
    "oil_temp_slope_C_per_h",
    "vib_roll_mean",
    "h2_roll_max",
)

FEATURE_NAMES: tuple[str, ...] = RAW_FEATURES + ENGINEERED_FEATURES

# Features on which the health score must be monotone DEcreasing (higher value
# -> same or lower score). Used as LightGBM `monotone_constraints` = -1.
MONOTONE_DECREASING: tuple[str, ...] = (
    "d_h2",
    "d_oil_temp",
    "d_load",
    "d_vib",
    "d_abs_resid",
    "d_volt",
    "volt_dev",
    "h2_rate_ppm_per_day",
    "oil_temp_C",
    "vibration_rms_ms2",
    "vib_roll_mean",
    "load_pu",
    "abs_in_out_resid",
)

# Physical-driver group of each feature, used to apply config.FEATURE_GAIN_MULTIPLIERS.
# Features that are not listed (ambient temperature, humidity) keep a multiplier of 1.0.
FEATURE_WEIGHT_GROUPS: dict[str, str] = {
    "output_voltage_V": "voltage",
    "volt_dev": "voltage",
    "d_volt": "voltage",
    "input_current_A": "current",
    "output_current_A": "current",
    "load_pu": "current",
    "d_load": "current",
    "in_out_resid": "current",
    "abs_in_out_resid": "current",
    "d_abs_resid": "current",
    "h2_ppm": "h2",
    "d_h2": "h2",
    "h2_rate_ppm_per_day": "h2",
    "h2_roll_max": "h2",
    "vibration_rms_ms2": "vibration",
    "vib_ratio": "vibration",
    "d_vib": "vibration",
    "vib_roll_mean": "vibration",
    "oil_temp_C": "oil_temperature",
    "oil_rise": "oil_temperature",
    "d_oil_temp": "oil_temperature",
    "oil_temp_slope_C_per_h": "oil_temperature",
}

# Trend features that are NaN until a full window is available.
_SLOPE_FEATURES: tuple[str, ...] = ("h2_rate_ppm_per_day", "oil_temp_slope_C_per_h")


def _self_check() -> None:
    if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise RuntimeError("Duplicate feature names")
    unknown = set(MONOTONE_DECREASING) - set(FEATURE_NAMES)
    if unknown:
        raise RuntimeError(f"Monotone constraints reference unknown features: {sorted(unknown)}")
    unknown = set(FEATURE_WEIGHT_GROUPS) - set(FEATURE_NAMES)
    if unknown:
        raise RuntimeError(f"Weight groups reference unknown features: {sorted(unknown)}")
    unknown = set(FEATURE_WEIGHT_GROUPS.values()) | set(cfg.FEATURE_GAIN_MULTIPLIERS)
    unknown -= set(FEATURE_WEIGHT_GROUPS.values()) & set(cfg.FEATURE_GAIN_MULTIPLIERS)
    if unknown:
        raise RuntimeError(f"Weight group / multiplier names do not match: {sorted(unknown)}")
    if any(m <= 0 for m in cfg.FEATURE_GAIN_MULTIPLIERS.values()):
        raise RuntimeError("Feature gain multipliers must be positive")
    assert_no_forbidden_columns(FEATURE_NAMES)  # raises LeakageError


_self_check()


def get_feature_names() -> list[str]:
    """Ordered feature names, identical to the columns returned by build_features."""
    return list(FEATURE_NAMES)


def get_monotone_constraints(feature_names: Sequence[str] | None = None) -> list[int]:
    """LightGBM `monotone_constraints` list: -1 for monotone-decreasing features, else 0."""
    names = list(FEATURE_NAMES if feature_names is None else feature_names)
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"Unknown feature names: {unknown}")
    mono = set(MONOTONE_DECREASING)
    return [-1 if n in mono else 0 for n in names]


def get_feature_contri(feature_names: Sequence[str] | None = None) -> list[float]:
    """LightGBM `feature_contri` list: the split-gain multiplier of each feature (1.0 if ungrouped)."""
    names = list(FEATURE_NAMES if feature_names is None else feature_names)
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"Unknown feature names: {unknown}")
    return [float(cfg.FEATURE_GAIN_MULTIPLIERS.get(FEATURE_WEIGHT_GROUPS.get(n, ""), 1.0)) for n in names]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _check_window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window!r}")


def _check_single_ordered_run(df: pd.DataFrame) -> None:
    if cfg.RUN_ID in df.columns and df[cfg.RUN_ID].nunique(dropna=False) > 1:
        raise ValueError("build_features works on ONE run; use build_features_by_run for several")
    if cfg.T_HOURS in df.columns and len(df) > 1:
        if not (df[cfg.T_HOURS].diff().dropna() > 0).all():
            raise ValueError("rows must be in strictly increasing time order (t_hours)")


def _norm_distance(x: pd.Series, limits: cfg.Limits) -> pd.Series:
    """0 at the Caution threshold, 1 at the Critical threshold, clipped to [0, DIST_CLIP_MAX]."""
    return ((x - limits.caution) / (limits.critical - limits.caution)).clip(0.0, cfg.DIST_CLIP_MAX)


def _trailing_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope (per sample) over the trailing `window` samples, current sample included.

    NaN until a full window exists, and NaN if any value in the window is NaN.
    """
    padded = np.concatenate([np.full(window - 1, np.nan), np.asarray(values, dtype="float64")])
    windows = sliding_window_view(padded, window)  # row t = samples t-window+1 .. t
    x = np.arange(window, dtype="float64")
    xc = x - x.mean()
    return windows @ (xc / np.dot(xc, xc))  # NaN propagates


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, window: int = cfg.FEATURE_WINDOW) -> pd.DataFrame:
    """Build the feature matrix for ONE run, using only past and current samples.

    `df` must contain the 8 sensor columns, ordered by time (oldest first). It may
    also contain meta/label columns; they are ignored (run_id and t_hours are only
    used for sanity checks). The input is not modified. Returns a float64
    DataFrame with the same index as `df` and columns == FEATURE_NAMES.
    """
    _check_window(window)
    missing = [c for c in cfg.INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")
    _check_single_ordered_run(df)

    x = df.loc[:, list(cfg.INPUT_FEATURES)].astype("float64")
    if x.empty:
        return pd.DataFrame(columns=list(FEATURE_NAMES), index=df.index, dtype="float64")

    i_in = x[cfg.INPUT_CURRENT_A]
    i_out = x[cfg.OUTPUT_CURRENT_A]
    volts = x[cfg.OUTPUT_VOLTAGE_V]
    ambient = x[cfg.AMBIENT_TEMP_C]
    vib = x[cfg.VIBRATION_RMS]
    h2 = x[cfg.H2_PPM]
    oil = x[cfg.OIL_TEMP_C]

    # Instantaneous (current-sample) features
    load_pu = i_out / cfg.RATED_LV_CURRENT_A
    oil_rise = oil - ambient

    expected_in = np.sqrt((i_out / cfg.TURNS_RATIO) ** 2 + cfg.NO_LOAD_CURRENT_A**2)  # always > 0
    in_out_resid = (i_in - expected_in) / expected_in
    abs_resid = in_out_resid.abs()

    vib_expected = cfg.VIB_COEF_VOLTAGE * (volts / cfg.LV_VOLTAGE_V) ** 2 + cfg.VIB_COEF_LOAD * load_pu**2
    vib_ratio = vib / vib_expected.where(vib_expected > 0)  # NaN, never inf, if undefined

    # Normalized distance to limit: 0 at Caution, 1 at Critical, capped at 1.5
    d_h2 = _norm_distance(h2, cfg.THRESHOLDS["h2_ppm"])
    d_oil = _norm_distance(oil, cfg.THRESHOLDS["oil_temp_C"])
    d_load = _norm_distance(load_pu, cfg.THRESHOLDS["load_pu"])
    d_vib = _norm_distance(vib, cfg.THRESHOLDS["vibration_rms_ms2"])
    d_resid = _norm_distance(abs_resid, cfg.THRESHOLDS["input_current_residual"])
    volt_dev = (volts - cfg.LV_VOLTAGE_V).abs() / cfg.LV_VOLTAGE_V
    d_volt = _norm_distance(volt_dev, cfg.THRESHOLDS["voltage_dev"])

    # Trend features over the trailing window
    h2_rate = _trailing_slope(h2.to_numpy(), window) * cfg.SAMPLES_PER_DAY  # ppm / day
    oil_slope = _trailing_slope(oil.to_numpy(), window) / cfg.SAMPLE_PERIOD_H  # degC / hour
    vib_roll_mean = vib.rolling(window, min_periods=1).mean()
    h2_roll_max = h2.rolling(window, min_periods=1).max()

    feats = {c: x[c] for c in cfg.INPUT_FEATURES}
    feats.update(
        {
            "load_pu": load_pu,
            "oil_rise": oil_rise,
            "in_out_resid": in_out_resid,
            "abs_in_out_resid": abs_resid,
            "vib_ratio": vib_ratio,
            "d_h2": d_h2,
            "d_oil_temp": d_oil,
            "d_load": d_load,
            "d_vib": d_vib,
            "d_abs_resid": d_resid,
            "volt_dev": volt_dev,
            "d_volt": d_volt,
            "h2_rate_ppm_per_day": pd.Series(h2_rate, index=df.index),
            "oil_temp_slope_C_per_h": pd.Series(oil_slope, index=df.index),
            "vib_roll_mean": vib_roll_mean,
            "h2_roll_max": h2_roll_max,
        }
    )
    out = pd.DataFrame(feats, index=df.index)[list(FEATURE_NAMES)]
    assert_no_forbidden_columns(out)
    return out


def build_features_by_run(df: pd.DataFrame, window: int = cfg.FEATURE_WINDOW) -> pd.DataFrame:
    """Apply build_features to each run_id separately (for training/evaluation).

    Rows inside each run must already be in time order (data.load_dataset sorts
    them). The result has the same index and row order as `df`; no window ever
    crosses a run boundary.
    """
    if cfg.RUN_ID not in df.columns:
        raise ValueError(f"'{cfg.RUN_ID}' column is required")
    if df[cfg.RUN_ID].isna().any():
        raise ValueError(f"'{cfg.RUN_ID}' contains NaN")
    _check_window(window)

    out = np.full((len(df), len(FEATURE_NAMES)), np.nan)
    for _, pos in df.groupby(cfg.RUN_ID, sort=False).indices.items():
        out[pos] = build_features(df.iloc[pos], window).to_numpy()
    return pd.DataFrame(out, index=df.index, columns=list(FEATURE_NAMES))


def warmup_mask(df: pd.DataFrame, window: int = cfg.FEATURE_WINDOW) -> pd.Series:
    """True for the first window-1 samples of each run (trend features are NaN there)."""
    _check_window(window)
    if cfg.RUN_ID in df.columns:
        position = df.groupby(cfg.RUN_ID, sort=False).cumcount()
    else:
        position = pd.Series(np.arange(len(df)), index=df.index)
    return position < (window - 1)
