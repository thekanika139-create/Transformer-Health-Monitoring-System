"""Build data/thms_dataset_voltage.csv: the original dataset made voltage-aware.

Why: in the original dataset the health score hardly depends on supply voltage, so no
model can give voltage a large weight. In the live demo the operator raises and lowers
the supply voltage by hand, so the score has to respond to it. This script

1. RELABELS every original row with a voltage rule: the voltage deviation |V - 415| / 415 is
   turned into a normalized distance (0 at config.THRESHOLDS["voltage_dev"].caution, 1 at its
   critical value) and a voltage score with the SAME severity curve as the formula baseline
   (config.BASELINE_SEVERITY_KNOTS). New score = min(original score, voltage score), so
   rows with healthy voltage keep their original label.
2. ADDS "voltage_excursion" runs: copies of NORMAL runs whose output voltage follows a
   random operator-style profile (ramps up and down, holds, returns to nominal). Vibration is
   rescaled with the same voltage dependence the features use, then the rows are labelled as in 1.
   Each copy stays in the split of its source run, so no run is shared between splits.

Run from the project root:
    python make_voltage_dataset.py                          # default: 1 copy of 35% of the normal runs
    python make_voltage_dataset.py --copies 1 --fraction 1.0   # a copy of every normal run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

OUT_PATH = cfg.PROJECT_ROOT / "data" / "thms_dataset_voltage.csv"
SOURCE_PATH = cfg.PROJECT_ROOT / "data" / "thms_dataset.csv"
V_MIN, V_MAX = 335.0, 495.0  # inside the validation range (300-500 V)


def voltage_score(volts: np.ndarray) -> np.ndarray:
    """Score (0-100) implied by the supply voltage alone."""
    lim = cfg.THRESHOLDS["voltage_dev"]
    dev = np.abs(np.asarray(volts, dtype="float64") - cfg.LV_VOLTAGE_V) / cfg.LV_VOLTAGE_V
    d = np.clip((dev - lim.caution) / (lim.critical - lim.caution), 0.0, cfg.DIST_CLIP_MAX)
    knots = np.asarray(cfg.BASELINE_SEVERITY_KNOTS, dtype="float64")
    return cfg.SCORE_MAX * (1.0 - np.interp(d, knots[:, 0], knots[:, 1]))


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the voltage rule to the score and re-derive state columns from the bands."""
    out = df.copy()
    score = np.minimum(out[cfg.HEALTH_SCORE].to_numpy(dtype="float64"), voltage_score(out[cfg.OUTPUT_VOLTAGE_V].to_numpy()))
    out[cfg.HEALTH_SCORE] = np.round(score, 1)
    e_normal, e_caution, e_warning = cfg.STATE_BAND_EDGES
    state = np.select([score > e_normal, score >= e_caution, score >= e_warning], [0, 1, 2], default=3)
    out[cfg.HEALTH_STATE] = state.astype("int64")
    out[cfg.HEALTH_STATE_NAME] = out[cfg.HEALTH_STATE].map(cfg.HEALTH_STATE_NAMES)
    return out


def voltage_profile(rng: np.random.Generator, n: int) -> np.ndarray:
    """Operator-style voltage: hold, ramp to a new level, hold, ... (2-12 h holds, 0.5-2 h ramps)."""
    nominal = cfg.LV_VOLTAGE_V
    knots_t, knots_v = [0], [nominal * (1 + rng.normal(0, 0.01))]
    t = 0
    while t < n:
        hold = int(rng.integers(8, 48))
        ramp = int(rng.integers(2, 8))
        if rng.random() < 0.25:  # back to roughly nominal
            target = nominal * (1 + rng.normal(0, 0.01))
        else:
            target = nominal * (1 + rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.22))
        target = float(np.clip(target, V_MIN, V_MAX))
        knots_t += [t + ramp, t + ramp + hold]
        knots_v += [target, target]
        t += ramp + hold
    v = np.interp(np.arange(n), knots_t, knots_v)
    return np.clip(v + rng.normal(0, 1.5, n), V_MIN, V_MAX)


def make_excursion_run(src: pd.DataFrame, new_id: int, rng: np.random.Generator) -> pd.DataFrame:
    run = src.copy()
    old_v = run[cfg.OUTPUT_VOLTAGE_V].to_numpy()
    new_v = voltage_profile(rng, len(run))
    load_pu = run[cfg.OUTPUT_CURRENT_A].to_numpy() / cfg.RATED_LV_CURRENT_A

    def expected(v: np.ndarray) -> np.ndarray:
        return cfg.VIB_COEF_VOLTAGE * (v / cfg.LV_VOLTAGE_V) ** 2 + cfg.VIB_COEF_LOAD * load_pu**2

    run[cfg.OUTPUT_VOLTAGE_V] = np.round(new_v, 1)
    run[cfg.VIBRATION_RMS] = np.round(np.clip(run[cfg.VIBRATION_RMS].to_numpy() * expected(new_v) / expected(old_v), 0, 20), 3)
    run[cfg.RUN_ID] = new_id
    run[cfg.FAULT_TYPE] = "voltage_excursion"
    return run


def build(copies: int, fraction: float = 0.35, source: Path = SOURCE_PATH, seed: int = cfg.RANDOM_SEED) -> pd.DataFrame:
    base = pd.read_csv(source).sort_values([cfg.RUN_ID, cfg.T_HOURS], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    next_id = int(base[cfg.RUN_ID].max()) + 1
    extra = []
    normal = base[base[cfg.FAULT_TYPE] == "normal"]
    for split_name in cfg.SPLIT_NAMES:  # take the same share of normal runs from every split
        ids = sorted(normal.loc[normal[cfg.SPLIT] == split_name, cfg.RUN_ID].unique())
        for rid in ids[: int(round(fraction * len(ids)))]:
            src = base[base[cfg.RUN_ID] == rid]
            for _ in range(copies):
                extra.append(make_excursion_run(src, next_id, rng))
                next_id += 1
    full = pd.concat([base, *extra], ignore_index=True) if extra else base
    full = relabel(full)
    return full.sort_values([cfg.RUN_ID, cfg.T_HOURS], kind="stable").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the voltage-aware THMS dataset.")
    ap.add_argument("--copies", type=int, default=1, help="voltage-excursion copies per selected normal run (default %(default)s)")
    ap.add_argument("--fraction", type=float, default=0.35, help="share of normal runs (per split) that get copies (default %(default)s)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    df = build(args.copies, args.fraction)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {df[cfg.RUN_ID].nunique()} runs")
    print(df.groupby([cfg.SPLIT, cfg.FAULT_TYPE]).size().unstack(fill_value=0).to_string())
    print(df.groupby(cfg.SPLIT)[cfg.HEALTH_STATE_NAME].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
