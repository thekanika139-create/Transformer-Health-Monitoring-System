"""Tiny settings file: only the load resistance used to CALCULATE the current shown on the dashboard.

current (A) = voltage (V) / load_resistance (ohm)      -- so the current rises and falls with the voltage.
The value is kept in settings.json next to this file and can be changed from the dashboard.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent / "settings.json"
DEFAULT_OHMS = 100.0          # your rheostat is 100 ohm
_lock = threading.Lock()


def get_load_resistance() -> float:
    with _lock:
        try:
            v = float(json.loads(PATH.read_text()).get("load_resistance_ohm", DEFAULT_OHMS))
            return v if v > 0 else DEFAULT_OHMS
        except Exception:
            return DEFAULT_OHMS


def set_load_resistance(ohms: float) -> float:
    ohms = float(ohms)
    if not (ohms > 0):
        raise ValueError("resistance must be greater than 0 ohm")
    with _lock:
        PATH.write_text(json.dumps({"load_resistance_ohm": ohms}, indent=2))
    return ohms
