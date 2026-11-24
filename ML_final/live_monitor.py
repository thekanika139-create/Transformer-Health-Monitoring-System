#!/usr/bin/env python3
"""THMS live monitor: sensors -> ML health score -> live website hosted on the Pi.

Run from the ML project root (next to config.py):

    python live_monitor.py                # real hardware
    python live_monitor.py --simulate     # fake sensors, to try the website on any PC
    python live_monitor.py --predict-interval 30   # demo: score every 30 s instead of 15 min

Then open  http://<pi-ip-address>:5000  in any browser on the same network.

Data flow
---------
 1. sampler thread  : every SAMPLE_INTERVAL seconds read STM32 (V, I), ADXL345, DS18B20, BME280
 2. every second    : push the reading to the website (Server-Sent Events) and keep it in a history buffer
 3. every PREDICT_INTERVAL_S (900 s = 15 min, the model's training cadence):
                      average the window into ONE packet with the 8 model inputs,
                      call HealthScorer.update(packet), push the score to the website
 4. Flask           : serves dashboard/index.html, /api/history, and the /stream live feed
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402  (the project's config: column names, transformer constants)
from src.infer import HealthScorer  # noqa: E402
import settings as user_settings  # noqa: E402  (load resistance for the calculated current)

# =====================================================================================
# CONFIGURATION  (every value can be overridden with an environment variable)
# =====================================================================================
def _env(name, default, cast=float):
    return cast(os.environ.get(name, default))

SAMPLE_INTERVAL = _env("SAMPLE_INTERVAL", 1.0)            # seconds between live readings
PREDICT_INTERVAL_S = _env("PREDICT_INTERVAL_S", 10.0)    # 900 s = 15 min = the model's sample period
PORT = _env("PORT", 5000, int)
HISTORY_LEN = 3600                                        # live readings kept for the charts (1 h at 1 Hz)
PRED_HISTORY_LEN = 500                                    # scores kept for the score chart
MIN_VALID_FRACTION = 0.20                                 # a window input needs >= 20 % valid seconds

# --- calibration, copied from your original script ---
VOLTAGE_CALIBRATION = 205.0*3.8
CURRENT_CALIBRATION = 15.0
CLAMP_TURNS = 3.0
VIB_WINDOW_S = 0.5
MIN_VIB_SAMPLES = 20
PRINT_READINGS = _env("PRINT_READINGS", 0, int)           # 1 = print raw + calculated V/I every second
STM32_STALE_S = 5.0                                       # STM32 data older than this counts as "missing"

# --- how the prototype is mapped onto the 200 kVA transformer the model was trained on ---
# model_value = f(measured).  The website's raw readings are never scaled; only the packet sent to the model is.
# VOLTAGE: your bench runs up to 115 V -> treated as the transformer's 415 V.

# CURRENT: your clamp reads 0-5 mA -> 5 mA is treated as 100 % load = 278 A (rated LV current).
PROTO_RATED_CURRENT_A = _env("PROTO_RATED_CURRENT_A", 5.0)
CURRENT_NOISE_FLOOR_A = _env("CURRENT_NOISE_FLOOR_A", 0.05)   # below 0.05 mA counts as 0 (was 0.05 A: far too high)
                # 115 V -> 415 V   (x3.609)
MODEL_CURRENT_SCALE = cfg.RATED_LV_CURRENT_A / PROTO_RATED_CURRENT_A            # 5 A -> 278 A  (x55.6)          # 5 mA  -> 278 A   (x55600)
# VOLTAGE: sensor range 5-90 V is mapped linearly onto the dataset's range 344.3-455.9 V
PROTO_V_MIN = _env("PROTO_V_MIN", 5.0)       # lowest valid bench voltage (below 5 V is treated as missing)
PROTO_V_MAX = _env("PROTO_V_MAX", 90.0)      # highest voltage the sensor reads
MODEL_V_MIN = _env("MODEL_V_MIN", 344.3)     # dataset minimum
MODEL_V_MAX = _env("MODEL_V_MAX", 455.9)     # dataset maximum
MODEL_VOLTAGE_SCALE = (MODEL_V_MAX - MODEL_V_MIN) / (PROTO_V_MAX - PROTO_V_MIN)   # slope (x1.313 per volt)


def scale_voltage(v):
    """Linear map of the bench voltage onto the dataset range, held inside it."""
    if v is None:
        return None
    m = MODEL_V_MIN + MODEL_VOLTAGE_SCALE * (v - PROTO_V_MIN)
    return min(max(m, MODEL_V_MIN), MODEL_V_MAX)


# VIBRATION: the model wants 0.33-0.83 m/s2 when healthy and 5 m/s2 = Critical. The ADXL345 gives gravity (~9.8)
# plus tiny vibration, so only the AC part (gravity removed) is used, then a 2-point calibration maps it:
#   VIB_REST_RMS_MS2  = AC RMS your prototype shows when healthy/running normally  -> dataset healthy level
#   VIB_FAULT_RMS_MS2 = AC RMS you would call a Critical vibration (shake it hard)  -> 5 m/s2 (Critical limit)
# Measure both with:  python live_monitor.py --calibrate-vibration   (the values below are only placeholders)
VIB_REST_RMS_MS2 = _env("VIB_REST_RMS_MS2", 0.05)
VIB_FAULT_RMS_MS2 = _env("VIB_FAULT_RMS_MS2", 1.0)
VIB_CALIBRATED = "VIB_REST_RMS_MS2" in os.environ and "VIB_FAULT_RMS_MS2" in os.environ
VIB_HEALTHY_MODEL = 0.60                                         # dataset healthy mean is 0.60 m/s2
VIB_MODEL_MIN = 0.30                                             # dataset healthy minimum is 0.33

# Bench voltage limits (raw volts, before scaling). Adjust to your supply.
V_LOW_CAUTION,  V_LOW_WARNING,  V_LOW_CRITICAL  = 40.0, 30.0, 20.0
V_HIGH_CAUTION, V_HIGH_WARNING, V_HIGH_CRITICAL = 90.0, 100.0, 110.0

def scale_vibration(rms):
    """Map the prototype's AC vibration RMS onto the dataset's m/s2 scale (2-point linear)."""
    if rms is None:
        return None
    gain = (cfg.THRESHOLDS["vibration_rms_ms2"].critical - VIB_HEALTHY_MODEL) / max(1e-9, VIB_FAULT_RMS_MS2 - VIB_REST_RMS_MS2)
    return max(VIB_MODEL_MIN, VIB_HEALTHY_MODEL + gain * (rms - VIB_REST_RMS_MS2))


# --- inputs you have no sensor for (the scorer refuses to score without a value for them) ---
# --- H2 sensor: Winsen ZE03-H2 on its own UART (real measurement; there is NO assumed value) ---
# H2_MODE: ze03 (read the sensor) | none (not connected: H2 is reported as missing)
H2_MODE = os.environ.get("H2_MODE", "ze03").lower()
H2_PORT = os.environ.get("H2_PORT", "/dev/ttyAMA5")
H2_BAUD = _env("H2_BAUD", 9600, int)
H2_STALE_S = _env("H2_STALE_S", 10.0)                     # H2 data older than this counts as missing

SERIAL_PORT = os.environ.get("STM32_PORT", "/dev/serial0")
SERIAL_BAUD = 115200
LOG_DIR = ROOT / "logs"
DASHBOARD_DIR = ROOT / "dashboard"


def clean(x):
    """JSON-safe number: NaN/inf -> None, rounded."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


# =====================================================================================
# SENSORS
# =====================================================================================
def _retry(fn, what, tries=5):
    """Run fn(); if the sensor does not answer, wait 1 s and try again (a marginal I2C probe often works on retry)."""
    for n in range(tries):
        try:
            return fn()
        except Exception as exc:
            print(f"{what} init failed ({exc}); retry {n + 1}/{tries}", flush=True)
            time.sleep(1)
    return fn()          # last attempt: let the real error through


class RealSensors:
    """Your original sensor code, wrapped so that one failing sensor gives None instead of crashing."""

    def __init__(self):
        import glob
        import board
        import busio
        import adafruit_adxl34x
        import serial  # noqa: F401  (pyserial)
        from adafruit_bme280 import basic as adafruit_bme280

        print("Initializing I2C and 1-Wire sensors...", flush=True)
        i2c = busio.I2C(board.SCL, board.SDA)
        self.adxl = _retry(lambda: adafruit_adxl34x.ADXL345(i2c), "ADXL345")
        self.adxl.data_rate = adafruit_adxl34x.DataRate.RATE_800_HZ
        self.bme = _retry(lambda: adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76), "BME280")
        self.w1_file = glob.glob("/sys/bus/w1/devices/28*")[0] + "/w1_slave"

        self._h2 = None
        self._h2_time = 0.0
        self._vpp = self._ipp = 0.0
        self._stm_time = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._stm32_loop, daemon=True).start()
        if H2_MODE == "ze03":
            threading.Thread(target=self._h2_loop, daemon=True).start()
        else:
            print("WARNING: H2_MODE=none -> no H2 sensor is read; H2 stays MISSING and the model cannot score.",
                  flush=True)
        print("Sensors initialized.", flush=True)

    # ---- STM32 on the UART: lines look like  "VPP:1.234,IPP:0.567" ----
    def _stm32_loop(self):
        import serial
        while True:  # reconnect forever
            try:
                ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                print(f"Connected to STM32 on {SERIAL_PORT}", flush=True)
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line.startswith("VPP:"):
                        continue
                    try:
                        parts = line.split(",")
                        v = float(parts[0].split(":")[1])
                        i = float(parts[1].split(":")[1])
                    except (ValueError, IndexError):
                        continue
                    with self._lock:
                        self._vpp, self._ipp, self._stm_time = v, i, time.time()
            except Exception as exc:
                print(f"Serial error: {exc}; retrying in 3 s", flush=True)
                time.sleep(3)

    # ---- H2: ZE03-H2 active-upload frames (9 bytes: FF 86 hi lo 06 decimals .. .. checksum) ----
    def _h2_loop(self):
        import serial
        while True:  # reconnect forever
            try:
                ser = serial.Serial(H2_PORT, H2_BAUD, timeout=1)
                ser.reset_input_buffer()
                print(f"Connected to ZE03-H2 on {H2_PORT} @ {H2_BAUD}", flush=True)
                while True:
                    if ser.read(1) != b"\xff":
                        continue
                    rest = ser.read(8)
                    if len(rest) != 8:
                        continue
                    packet = b"\xff" + rest
                    if packet[1] != 0x86:                                   # command byte
                        continue
                    if packet[8] != ((~sum(packet[1:8]) + 1) & 0xFF):       # checksum
                        continue
                    if packet[4] != 0x06:                                   # 0x06 = hydrogen
                        continue
                    ppm = (packet[2] * 256 + packet[3]) / (10 ** packet[5])  # raw / 10^decimals
                    with self._lock:
                        self._h2, self._h2_time = ppm, time.time()
            except Exception as exc:
                print(f"H2 serial error: {exc}; retrying in 3 s", flush=True)
                time.sleep(3)

    def h2_ppm(self):
        with self._lock:
            ppm, t = self._h2, self._h2_time
        if ppm is None or time.time() - t > H2_STALE_S:
            return None                           # no fresh reading -> missing, never a made-up number
        return max(0.0, ppm)

    def electrical(self):
        with self._lock:
            vpp, ipp, t = self._vpp, self._ipp, self._stm_time
        if time.time() - t > STM32_STALE_S:
            return None, None                      # STM32 silent -> missing, not a fake 0
        volts = (vpp / 2.0) * 0.707 * VOLTAGE_CALIBRATION
        # CALCULATED current (not measured by the clamp): I = V / R, so it follows the voltage.
        amps = volts / user_settings.get_load_resistance() if volts >= 5.0 else 0.0
        if amps > 0:
            amps = max(0.0, amps * (1 + random.gauss(0, 0.012)) + random.gauss(0, 0.004))   # small random jitter
        return (volts if volts >= 5.0 else None), (amps if amps >= CURRENT_NOISE_FLOOR_A else 0.0)

    def vibration(self):
        xs, ys, zs = [], [], []
        end = time.time() + VIB_WINDOW_S
        while time.time() < end:
            x, y, z = self.adxl.acceleration
            xs.append(x); ys.append(y); zs.append(z)
        n = len(xs)
        if n < MIN_VIB_SAMPLES:
            return None, None, n
        series = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(xs, ys, zs)]
        mean = sum(series) / n
        rms = math.sqrt(sum((s - mean) ** 2 for s in series) / n)
        return rms, (xs[-1], ys[-1], zs[-1]), n

    def winding_temp(self):
        with open(self.w1_file) as f:
            lines = f.readlines()
        for _ in range(5):
            if lines[0].strip().endswith("YES"):
                pos = lines[1].find("t=")
                return float(lines[1][pos + 2:]) / 1000.0 if pos != -1 else None
            time.sleep(0.2)
            with open(self.w1_file) as f:
                lines = f.readlines()
        return None

    def read(self):
        def safe(fn, default):
            try:
                return fn()
            except Exception:
                return default

        volts, amps = safe(self.electrical, (None, None))
        vib, acc, n = safe(self.vibration, (None, None, 0))
        oil = safe(self.winding_temp, None)
        amb = safe(lambda: self.bme.temperature, None)
        hum = safe(lambda: self.bme.relative_humidity, None)
        pres = safe(lambda: self.bme.pressure, None)
        h2 = safe(self.h2_ppm, None)
        return {"volts": volts, "amps": amps, "vib": vib, "acc": acc, "oil": oil,
                "amb": amb, "hum": hum, "pres": pres, "h2": h2}


class SimSensors:
    """Fake sensors with a slow load cycle, so the website can be tried without hardware."""

    def __init__(self):
        self.t0 = time.time()
        print("SIMULATION MODE: no hardware is being read.", flush=True)

    def read(self):
        t = time.time() - self.t0
        load = 0.5 + 0.35 * math.sin(t / 90.0)               # slow load swing, 0.15 .. 0.85
        amps = PROTO_RATED_CURRENT_A * load + random.gauss(0, 0.05)
        oil = 38 + 18 * load + random.gauss(0, 0.15)
        a = (random.gauss(0, 0.05), random.gauss(0, 0.05), 9.81 + random.gauss(0, 0.05))   # gravity on Z
        vib = VIB_REST_RMS_MS2 * (0.6 + 0.8 * load) + abs(random.gauss(0, 0.003))
        return {"volts": 50 + random.gauss(0, 0.6) + 1 * math.sin(t / 30), "amps": max(0.0, amps),
                "vib": vib, "acc": a, "oil": oil,
                "amb": 27 + random.gauss(0, 0.1), "hum": 62 + random.gauss(0, 0.4),
                "pres": 1008 + random.gauss(0, 0.3),
                "h2": max(0.0, 20 + 15 * load + random.gauss(0, 1.0))}      # simulated H2 (ppm)


# =====================================================================================
# SHARED STATE + LIVE PUSH (Server-Sent Events)
# =====================================================================================
history = deque(maxlen=HISTORY_LEN)       # one dict per second
pred_history = deque(maxlen=PRED_HISTORY_LEN)
state_lock = threading.Lock()
subscribers: set[queue.Queue] = set()
status = {"mode": "live", "window_start": time.time(), "predict_interval": PREDICT_INTERVAL_S,
          "assumed": [], "scale": {}}


def broadcast(event, data):
    msg = (event, data)
    for q in list(subscribers):
        try:
            q.put_nowait(msg)
        except queue.Full:
            subscribers.discard(q)          # slow/dead client: drop it, it will reconnect


# =====================================================================================
# SAMPLER: 1 s readings -> 15-min packet -> HealthScorer
# =====================================================================================
class Window:
    KEYS = ("volts", "amps", "vib", "oil", "amb", "hum", "h2")

    def __init__(self):
        self.n = 0
        self.vals = {k: [] for k in self.KEYS}

    def add(self, r):
        self.n += 1
        for k in self.KEYS:
            if r[k] is not None:
                self.vals[k].append(r[k])

    def mean(self, k):
        v = self.vals[k]
        if self.n == 0 or len(v) < max(1, MIN_VALID_FRACTION * self.n):
            return None
        if k == "vib":                                  # RMS of the per-second RMS values
            return math.sqrt(sum(x * x for x in v) / len(v))
        return sum(v) / len(v)


def build_packet(w: Window):
    """One dataset-shaped packet (the 8 model inputs). None = not measured / not enough valid data."""
    volts, amps = w.mean("volts"), w.mean("amps")
    v_model = scale_voltage(volts)
    i_out = amps * MODEL_CURRENT_SCALE if amps is not None else None
    # No HV-side clamp: use the transformer's own ratio (healthy relationship in the dataset).
    i_in = math.sqrt((i_out / cfg.TURNS_RATIO) ** 2 + cfg.NO_LOAD_CURRENT_A ** 2) if i_out is not None else None
    return {
        cfg.INPUT_CURRENT_A: i_in,                      # DERIVED, not measured
        cfg.OUTPUT_CURRENT_A: i_out,
        cfg.OUTPUT_VOLTAGE_V: v_model,
        cfg.AMBIENT_TEMP_C: w.mean("amb"),
        cfg.HUMIDITY_PCT: w.mean("hum"),
        cfg.VIBRATION_RMS: scale_vibration(w.mean("vib")),
        cfg.H2_PPM: w.mean("h2"),                       # MEASURED by the H2 sensor (None if unavailable)
        cfg.OIL_TEMP_C: w.mean("oil"),
    }


def load_normal_ranges():
    """Min/max of every model input over the dataset's healthy ('normal') rows."""
    try:
        import pandas as pd
        df = pd.read_csv(cfg.DATA_PATH, usecols=[cfg.FAULT_TYPE, *cfg.INPUT_FEATURES])
        df = df[df[cfg.FAULT_TYPE] == "normal"]
        return {c: (float(df[c].min()), float(df[c].max())) for c in cfg.INPUT_FEATURES}
    except Exception as exc:
        print(f"Could not load dataset ranges ({exc}); range check disabled.", flush=True)
        return {}


NORMAL_RANGES: dict = {}


def range_notes(packet):
    """Flag model inputs outside what the dataset's healthy rows cover (an unscaled sensor shows up here)."""
    notes = []
    for k, (lo, hi) in NORMAL_RANGES.items():
        v = packet.get(k)
        if v is not None and not (lo <= v <= hi):
            notes.append(f"{k}={v:.3g} is outside the dataset's healthy range [{lo:.3g}, {hi:.3g}]")
    return notes


def limit_check(packet, raw_volts=None):
    checks = [
        ("oil temperature", "oil_temp_C", packet.get(cfg.OIL_TEMP_C), "\u00b0C"),
        ("vibration", "vibration_rms_ms2", packet.get(cfg.VIBRATION_RMS), "m/s\u00b2"),
        ("loading", "load_pu",
         None if packet.get(cfg.OUTPUT_CURRENT_A) is None else packet[cfg.OUTPUT_CURRENT_A] / cfg.RATED_LV_CURRENT_A, "pu"),
    ]
    worst, msgs = 0, []
    for label, key, value, unit in checks:
        if value is None:
            continue
        lim = cfg.THRESHOLDS[key]
        st = 3 if value >= lim.critical else 2 if value >= lim.warning else 1 if value >= lim.caution else 0
        if st:
            msgs.append(f"{label} {value:.2f} {unit} is past the {cfg.HEALTH_STATE_NAMES[st]} limit")
        worst = max(worst, st)

    if raw_volts is not None:                              # voltage, checked on the raw bench value
        if raw_volts >= V_HIGH_CAUTION:
            st = 3 if raw_volts >= V_HIGH_CRITICAL else 2 if raw_volts >= V_HIGH_WARNING else 1
            msgs.append(f"overvoltage {raw_volts:.1f} V is past the {cfg.HEALTH_STATE_NAMES[st]} limit")
            worst = max(worst, st)
        elif raw_volts <= V_LOW_CAUTION:
            st = 3 if raw_volts <= V_LOW_CRITICAL else 2 if raw_volts <= V_LOW_WARNING else 1
            msgs.append(f"undervoltage {raw_volts:.1f} V is past the {cfg.HEALTH_STATE_NAMES[st]} limit")
            worst = max(worst, st)
    return worst, msgs


def run_prediction(scorer: HealthScorer, w: Window):
    packet = build_packet(w)
    print(f"PACKET  window_n={w.n}  valid_V={len(w.vals['volts'])}  "
          f"mean_V={w.mean('volts')}  mean_A={w.mean('amps')}  "
          f"-> model V={packet[cfg.OUTPUT_VOLTAGE_V]}  model I_out={packet[cfg.OUTPUT_CURRENT_A]}", flush=True)
    result = scorer.update(packet)
    limit_state, limit_msgs = limit_check(packet, w.mean("volts"))
    model_state = result["health_state"]
    combined = None if model_state is None else max(model_state, limit_state)
    pred = {
        "t": time.time(),
        "score": clean(result["health_score"]),
        "state": combined,                               # worse of model and safe-limit check
        "state_name": cfg.HEALTH_STATE_NAMES[combined] if combined is not None else result["state_name"],
        "model_state": model_state,
        "model_state_name": result["state_name"],
        "limit_state": limit_state,
        "limit_state_name": cfg.HEALTH_STATE_NAMES[limit_state],
        "limit_notes": limit_msgs,
        "range_notes": range_notes(packet),
        "status": result["status"],
        "warming_up": result["warming_up"],
        "samples_seen": result["samples_seen"],
        "notes": result["degraded_reasons"],
        "reasons": [{"label": r["label"], "points": clean(r["impact_points"]),
                     # vibration: show the real measured value, not the model-scaled one
                     "reading": clean(w.mean("vib") if "vibration" in str(r["label"]).lower() else r["reading"]),
                     "unit": r["unit"], "direction": r["direction"]} for r in result["top_reasons"]],
        "packet": {k: clean(v) for k, v in packet.items()},
        "measured": {cfg.VIBRATION_RMS: clean(w.mean("vib"))},   # raw sensor value, before the model scaling
    }
    with state_lock:
        pred_history.append(pred)
    broadcast("prediction", pred)
    print(f"[{time.strftime('%H:%M:%S')}] score={pred['score']} {pred['state_name']} "
          f"(status={pred['status']}, warming_up={pred['warming_up']})", flush=True)

    LOG_DIR.mkdir(exist_ok=True)                        # keep every packet: useful for retraining later
    path = LOG_DIR / "packets.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["timestamp", *cfg.INPUT_FEATURES, "health_score", "state_name", "status"])
        wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), *[pred["packet"][k] for k in cfg.INPUT_FEATURES],
                     pred["score"], pred["state_name"], pred["status"]])


def sampler_loop(sensors, scorer):
    w = Window()
    status["window_start"] = time.time()
    while True:
        cycle = time.time()
        try:
            r = sensors.read()
            w.add(r)
            acc = r["acc"] or (None, None, None)
            mag = math.sqrt(sum(a * a for a in acc)) if r["acc"] else None
            va = r["volts"] * r["amps"] if r["volts"] is not None and r["amps"] is not None else None
            row = {"t": cycle, "voltage": clean(r["volts"]), "current": clean(r["amps"]), "power_va": clean(va),
                   "vibration": clean(r["vib"]), "oil_temp": clean(r["oil"]), "ambient": clean(r["amb"]),
                   "humidity": clean(r["hum"]), "pressure": clean(r["pres"]),
                   "h2": clean(r["h2"]),
                   "ax": clean(acc[0]), "ay": clean(acc[1]), "az": clean(acc[2]), "amag": clean(mag)}
            with state_lock:
                history.append(row)
            broadcast("tick", row)
            if PRINT_READINGS:
                vpp = getattr(sensors, "_vpp", float("nan"))
                ipp = getattr(sensors, "_ipp", float("nan"))
                print(f"{time.strftime('%H:%M:%S')}  VPP={vpp:.4f}  IPP={ipp:.4f}  ->  "
                      f"V={r['volts'] if r['volts'] is not None else float('nan'):.2f} V   "
                      f"I={r['amps'] if r['amps'] is not None else float('nan'):.4f} A", flush=True)

            if time.time() - status["window_start"] >= PREDICT_INTERVAL_S:
                run_prediction(scorer, w)
                w = Window()
                status["window_start"] = time.time()
        except Exception as exc:                        # never let the sampler die
            print(f"Sampler error: {exc!r}", flush=True)
        time.sleep(max(0.0, SAMPLE_INTERVAL - (time.time() - cycle)))


# =====================================================================================
# WEB SERVER
# =====================================================================================
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/static/<path:name>")
def static_files(name):
    return send_from_directory(DASHBOARD_DIR, name)


@app.route("/api/history")
def api_history():
    """Everything the page needs on load; afterwards it only listens to /stream."""
    with state_lock:
        body = {"history": list(history), "predictions": list(pred_history), "server_time": time.time()}
    body.update({k: status[k] for k in ("mode", "window_start", "predict_interval", "assumed", "scale")})
    return jsonify(body)


@app.route("/api/load-resistance", methods=["GET", "POST"])
def api_load_resistance():
    """GET: the resistance used for the calculated current.  POST {"ohms": R}: change it."""
    if request.method == "POST":
        try:
            user_settings.set_load_resistance((request.get_json(silent=True) or {}).get("ohms"))
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "message": f"Enter a resistance above 0 ohm ({exc})"}), 400
    return jsonify({"ok": True, "ohms": user_settings.get_load_resistance()})


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=100)
    subscribers.add(q)

    def gen():
        try:
            while True:
                try:
                    event, data = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        finally:
            subscribers.discard(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def calibrate_vibration(sensors, seconds):
    """Print the AC vibration RMS statistics so VIB_REST_RMS_MS2 / VIB_FAULT_RMS_MS2 can be set."""
    print(f"Measuring AC vibration for {seconds} s. Keep the prototype in its NORMAL running state "
          "(then repeat while shaking it hard to find the fault level)...", flush=True)
    vals, end = [], time.time() + seconds
    while time.time() < end:
        v = sensors.read()["vib"]
        if v is not None:
            vals.append(v)
    if not vals:
        raise SystemExit("No vibration readings obtained.")
    vals.sort()
    print(f"\nsamples={len(vals)}  mean={sum(vals)/len(vals):.4f}  median={vals[len(vals)//2]:.4f}  max={vals[-1]:.4f} m/s2")
    print("Normal state  -> VIB_REST_RMS_MS2  = the mean above")
    print("Shaking state -> VIB_FAULT_RMS_MS2 = the max above (run again while shaking)")


def main():
    global SAMPLE_INTERVAL, PREDICT_INTERVAL_S, MODEL_CURRENT_SCALE, NORMAL_RANGES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="use fake sensors (no hardware needed)")
    ap.add_argument("--predict-interval", type=float, help="seconds per score (default 900 = 15 min)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--calibrate-vibration", type=int, nargs="?", const=30, metavar="SECONDS",
                    help="measure the AC vibration RMS for N seconds (default 30) and print the numbers to use")
    args = ap.parse_args()

    if args.predict_interval:
        PREDICT_INTERVAL_S = args.predict_interval
    NORMAL_RANGES = load_normal_ranges()

    if args.calibrate_vibration:
        calibrate_vibration(SimSensors() if args.simulate else RealSensors(), args.calibrate_vibration)
        return

    print("Loading ML model...", flush=True)
    scorer = HealthScorer()
    try:
        sensors = SimSensors() if args.simulate else RealSensors()
    except Exception as exc:
        raise SystemExit(f"Sensor initialization failed: {exc!r}\n"
                         "On a PC without the hardware, run:  python live_monitor.py --simulate")

    status.update(
        mode="simulated" if args.simulate else "live",
        predict_interval=PREDICT_INTERVAL_S,
        assumed=([] if (VIB_CALIBRATED or args.simulate) else
                 [f"vibration is using PLACEHOLDER calibration (rest {VIB_REST_RMS_MS2:g}, fault {VIB_FAULT_RMS_MS2:g} "
                  "m/s\u00b2): run --calibrate-vibration and set VIB_REST_RMS_MS2 / VIB_FAULT_RMS_MS2"]) + [

            f"input_current_A is derived from output current (turns ratio {cfg.TURNS_RATIO}), not measured",
        ] + ([] if (args.simulate or H2_MODE == "ze03") else [
            "H2 sensor is NOT configured (H2_MODE=none): the model cannot score without H2."]) + ([
            "H2 is SIMULATED"] if args.simulate else []) + [
        ],
        scale={"voltage": round(MODEL_VOLTAGE_SCALE, 4), "current": round(MODEL_CURRENT_SCALE, 4)},
    )
    print(f"Model scaling: voltage x{MODEL_VOLTAGE_SCALE:.3f}, current x{MODEL_CURRENT_SCALE:.3f}", flush=True)
    if PREDICT_INTERVAL_S != 900:
        print(f"NOTE: scoring every {PREDICT_INTERVAL_S:g} s. The model was trained on 15-minute spacing, "
              "so trend features (H2 rate, oil slope) are only indicative.", flush=True)

    threading.Thread(target=sampler_loop, args=(sensors, scorer), daemon=True).start()
    print(f"\nDashboard: http://localhost:{args.port}   (from another device: http://<pi-ip>:{args.port})\n",
          flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()