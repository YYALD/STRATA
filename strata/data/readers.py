"""Readers of measured profiles: CSV matrices (encoding, delimiter, orientation, time unit and depth order detected
per file) and survey line files (borehole, airborne, ground) with their gate times."""

from __future__ import annotations

import copy
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common import file_sha256, sanitize_path


# The measured file is the only time / depth standard of the run:
#   * time  : the generation grid spans exactly [t_first, t_last] of the measured gates and
#             contains every measured gate time exactly (embedded grid, target_len >= gates);
#   * depth : the generated profiles live on the measured depth stations (spacing pattern kept);
#   * level : the measured decay defines the background; simulated data enter as a structural
#             prior registered onto the measured depth axis.
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16", "big5", "latin-1")
_DEPTH_NAME_HINTS = ("depth", "深度", "station", "测点", "孔深", "dep", "井深", "md")
_TIME_NAME_HINTS = ("time", "时间", "gate", "时间道", "channel")


@dataclass
class ProfileMatrix:
    path: str
    role: str
    depth: np.ndarray            # (D,) metres, strictly increasing
    time_ms: np.ndarray          # (n,) milliseconds, strictly increasing
    values: np.ndarray           # (D, n) float64
    orientation: str             # "rows=depth" | "rows=time"
    time_unit: str
    time_unit_reason: str
    orientation_reason: str
    axis_name: str
    encoding: str
    sha16: str
    notes: List[str] = field(default_factory=list)
    # how the file was read: format "matrix" (header row / column) or "survey_lines" (airborne / ground line file with
    # geometry columns), the line, gate window and stacking of a line file, the sensor-height column ...
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        d, t = self.depth, self.time_ms
        dd = np.diff(d) if d.size > 1 else np.zeros(0)
        out = {
            "path": self.path, "role": self.role, "sha256_16": self.sha16, "encoding": self.encoding,
            "axis_name": self.axis_name, "orientation": self.orientation,
            "orientation_reason": self.orientation_reason,
            "time_unit_detected": self.time_unit, "time_unit_reason": self.time_unit_reason,
            "n_depth": int(d.size), "depth_min_m": float(d.min()), "depth_max_m": float(d.max()),
            "depth_spacing_m": {"min": float(dd.min()) if dd.size else 0.0,
                                "median": float(np.median(dd)) if dd.size else 0.0,
                                "max": float(dd.max()) if dd.size else 0.0},
            "n_time": int(t.size), "time_min_ms": float(t.min()), "time_max_ms": float(t.max()),
            "value_min": float(np.nanmin(self.values)), "value_max": float(np.nanmax(self.values)),
            "nonpositive_count": int(np.sum(self.values <= 0)), "notes": list(self.notes),
        }
        if self.meta:
            out["read_meta"] = dict(self.meta)
        return out


def _detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    for enc in _ENCODINGS:
        try:
            head.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def _detect_delimiter(line: str) -> str:
    counts = {d: line.count(d) for d in (",", "\t", ";")}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return r"\s+"


def _parse_axis_token(tok: Any) -> float:
    s = str(tok).strip().strip('"').strip("'")
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        pass
    num = ""
    for ch in s:  # tolerate "0.054ms" / "10.0m"
        if ch.isdigit() or ch in ".-+eE":
            num += ch
        elif num:
            break
    try:
        return float(num) if num not in ("", "-", "+", ".") else float("nan")
    except Exception:
        return float("nan")


def _axis_stats(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    fin = v[np.isfinite(v)]
    out = {"n": float(v.size), "finite_frac": float(fin.size / max(v.size, 1)),
           "positive": float(fin.size > 0 and np.all(fin > 0)),
           "monotone": 0.0, "lin_cv": float("nan"), "log_cv": float("nan"), "dyn": float("nan"),
           "consecutive_int": 0.0}
    if fin.size >= 3:
        s = np.sort(fin)
        dif = np.diff(s)
        mono_inc = bool(np.all(np.diff(fin) > 0))
        mono_dec = bool(np.all(np.diff(fin) < 0))
        out["monotone"] = float(mono_inc or mono_dec)
        out["lin_cv"] = float(np.std(dif) / max(abs(np.mean(dif)), 1e-30))
        if np.all(s > 0):
            lr = np.diff(np.log(s))
            out["log_cv"] = float(np.std(lr) / max(abs(np.mean(lr)), 1e-30))
            out["dyn"] = float(s[-1] / max(s[0], 1e-30))
        out["consecutive_int"] = float(np.all(np.abs(fin - np.round(fin)) < 1e-9) and np.all(np.abs(dif - 1.0) < 1e-9))
    return out


def _time_likeness(st: Dict[str, float]) -> float:
    """Score how much an axis looks like a TEM gate-time axis (positive, log-spaced)."""
    if st["finite_frac"] < 0.5 or st["n"] < 3:
        return -9.0
    score = 0.0
    score += 2.0 if st["positive"] > 0 else -2.0
    score += 1.0 if st["monotone"] > 0 else -1.0
    if np.isfinite(st["lin_cv"]) and st["lin_cv"] > 0.25:
        score += 2.0      # non-uniform linear spacing (quasi-log gate grid)
    if np.isfinite(st["log_cv"]) and st["log_cv"] < 0.5:
        score += 1.0      # roughly constant ratio between gates
    if np.isfinite(st["dyn"]) and st["dyn"] >= 20.0:
        score += 1.0      # TEM windows span >= 1.3 decades
    if st["consecutive_int"] > 0:
        score -= 4.0      # 1..N = trace / station id, not a time axis
    return score


def _time_unit_to_ms(t: np.ndarray, forced: str, name_hint: str) -> Tuple[float, str, str]:
    """Return (factor_to_ms, unit, reason)."""
    forced = (forced or "auto").lower()
    table = {"s": 1000.0, "ms": 1.0, "us": 1.0e-3}
    if forced in table:
        return table[forced], forced, "forced by flag"
    nm = (name_hint or "").lower()
    if any(k in nm for k in ("time_s", "(s)", "[s]", "_s ", "sec")):
        return 1000.0, "s", f"axis name '{name_hint}'"
    if "ms" in nm or "毫秒" in nm:
        return 1.0, "ms", f"axis name '{name_hint}'"
    if any(k in nm for k in ("us", "μs", "µs", "微秒")):
        return 1.0e-3, "us", f"axis name '{name_hint}'"
    tmax = float(np.nanmax(t))
    if tmax <= 0.5:
        return 1000.0, "s", f"magnitude rule: t_max={tmax:.6g} <= 0.5 -> seconds"
    if tmax <= 2000.0:
        return 1.0, "ms", f"magnitude rule: 0.5 < t_max={tmax:.6g} <= 2000 -> milliseconds"
    return 1.0e-3, "us", f"magnitude rule: t_max={tmax:.6g} > 2000 -> microseconds"


def _fill_nonfinite_rows(t_ms: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fill NaN/inf per curve by log-time linear interpolation (edges: nearest valid)."""
    V = np.array(V, dtype=np.float64, copy=True)
    bad = ~np.isfinite(V)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return V, 0
    lt = np.log10(np.maximum(t_ms, 1e-30))
    for i in np.where(bad.any(axis=1))[0]:
        ok = ~bad[i]
        if ok.sum() >= 2:
            V[i, ~ok] = np.interp(lt[~ok], lt[ok], V[i, ok])
        elif ok.sum() == 1:
            V[i, ~ok] = V[i, ok][0]
    return V, n_bad


def _average_rows(V: np.ndarray, inv: np.ndarray, n: int) -> np.ndarray:
    """Mean of the rows sharing a position (inv = output row of each input row) over their FINITE samples only."""
    V = np.asarray(V, dtype=np.float64)
    fin = np.isfinite(V)
    s = np.zeros((n, V.shape[1]))
    c = np.zeros((n, V.shape[1]))
    np.add.at(s, inv, np.where(fin, V, 0.0))
    np.add.at(c, inv, fin.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def read_profile_matrix(path: str, role: str, read_by: str = "auto", time_unit: str = "auto",
                        verbose: bool = True) -> ProfileMatrix:
    """Read a header-row / header-column TEM matrix of any orientation into (depth x time)."""
    path = sanitize_path(path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"[DATA] {role} file does not exist: {path}")
    enc = _detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
    delim = _detect_delimiter(header_line)
    head_cells = [c for c in (header_line.split(delim) if delim != r"\s+" else header_line.split())]
    axis_name = str(head_cells[0]).strip().lstrip("\ufeff") if head_cells else ""
    hdr = np.array([_parse_axis_token(c) for c in head_cells[1:]], dtype=np.float64)
    body = pd.read_csv(path, header=None, skiprows=1, sep=delim if delim != r"\s+" else r"\s+",
                       encoding=enc, engine="c" if delim != r"\s+" else "python")
    col0 = pd.to_numeric(body.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    mat = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if mat.shape[1] != hdr.size:
        m = min(mat.shape[1], hdr.size)
        mat, hdr = mat[:, :m], hdr[:m]
    notes: List[str] = []

    # ---- orientation: which axis is the time axis ----
    nm = axis_name.lower()
    st_row, st_col = _axis_stats(hdr), _axis_stats(col0)
    rb = (read_by or "auto").lower()
    if rb == "row":
        rows_are, reason = "depth", "forced --read_by row (header row = time)"
    elif rb == "column":  # first COLUMN = time axis
        rows_are, reason = "time", "forced --read_by column (first column = time)"
    elif any(h in nm for h in _DEPTH_NAME_HINTS):
        rows_are, reason = "depth", f"first-column name '{axis_name}' is a depth/station name"
    elif any(h in nm for h in _TIME_NAME_HINTS):
        rows_are, reason = "time", f"first-column name '{axis_name}' is a time name"
    else:
        s_row, s_col = _time_likeness(st_row), _time_likeness(st_col)
        if s_row > s_col:
            rows_are = "depth"
        elif s_col > s_row:
            rows_are = "time"
        else:
            rows_are = "depth" if hdr.size <= col0.size else "time"
        reason = f"time-likeness header-row={s_row:.1f} first-col={s_col:.1f}"
    if rows_are == "depth":
        t_raw, d_raw, V = hdr, col0, mat
        orientation = "rows=depth"
    else:
        t_raw, d_raw, V = col0, hdr, mat.T
        orientation = "rows=time"
    t_name_hint = axis_name if rows_are == "time" else ""

    # ---- drop unusable time columns / depth rows ----
    t_ok = np.isfinite(t_raw) & (t_raw > 0)
    if not np.all(t_ok):
        notes.append(f"dropped {int((~t_ok).sum())} time columns with invalid axis values")
    d_ok = np.isfinite(d_raw)
    if not np.all(d_ok):
        notes.append(f"dropped {int((~d_ok).sum())} rows with invalid depth values")
    t_raw, d_raw = t_raw[t_ok], d_raw[d_ok]
    V = V[np.ix_(d_ok, t_ok)]
    fin_row = np.isfinite(V).mean(axis=1) >= 0.5
    if not np.all(fin_row):
        notes.append(f"dropped {int((~fin_row).sum())} curves with <50% finite values")
        V, d_raw = V[fin_row], d_raw[fin_row]
    if V.shape[0] < 2 or V.shape[1] < 3:
        raise ValueError(f"[DATA] {role} matrix too small after cleaning: {V.shape}")

    # ---- units ----
    fac, unit, unit_reason = _time_unit_to_ms(t_raw, time_unit, t_name_hint)
    t_ms = t_raw * fac
    dep_fac = 1000.0 if ("km" in nm and rows_are == "depth") else 1.0
    d_m = d_raw * dep_fac

    # ---- sort / de-duplicate ----
    ti = np.argsort(t_ms, kind="mergesort")
    t_ms, V = t_ms[ti], V[:, ti]
    if np.any(np.diff(t_ms) <= 0):
        ut, inv = np.unique(t_ms, return_inverse=True)
        Vn = np.zeros((V.shape[0], ut.size))
        cnt = np.bincount(inv, minlength=ut.size).astype(np.float64)
        for j in range(V.shape[1]):
            Vn[:, inv[j]] += V[:, j]
        V, t_ms = Vn / cnt[None, :], ut
        notes.append("duplicate time columns averaged")
    di = np.argsort(d_m, kind="mergesort")
    if np.any(di != np.arange(di.size)):
        notes.append("rows re-ordered by increasing depth")
    d_m, V = d_m[di], V[di]
    if np.any(np.diff(d_m) <= 0):
        ud, inv = np.unique(d_m, return_inverse=True)
        V, d_m = _average_rows(V, inv, ud.size), ud
        notes.append("duplicate depth rows averaged")
    V, n_fill = _fill_nonfinite_rows(t_ms, V)
    if n_fill:
        notes.append(f"filled {n_fill} non-finite samples by log-time interpolation")
    pm = ProfileMatrix(path=path, role=role, depth=d_m.astype(np.float64), time_ms=t_ms.astype(np.float64),
                       values=V.astype(np.float64), orientation=orientation, time_unit=unit,
                       time_unit_reason=unit_reason, orientation_reason=reason, axis_name=axis_name,
                       encoding=enc, sha16=file_sha256(path)[:16], notes=notes)
    if verbose:
        s = pm.summary()
        print("=" * 88)
        print(f"[DATA-READ] {role}: {path}")
        print(f"[DATA-READ]   encoding={enc} orientation={orientation} ({reason})")
        print(f"[DATA-READ]   time unit={unit} ({unit_reason}) -> {s['n_time']} gates "
              f"{s['time_min_ms']:.6g} .. {s['time_max_ms']:.6g} ms")
        print(f"[DATA-READ]   depth {s['n_depth']} rows {s['depth_min_m']:.6g} .. {s['depth_max_m']:.6g} m "
              f"(spacing min/med/max {s['depth_spacing_m']['min']:.4g}/{s['depth_spacing_m']['median']:.4g}/"
              f"{s['depth_spacing_m']['max']:.4g})")
        print(f"[DATA-READ]   values {s['value_min']:.4g} .. {s['value_max']:.4g} | non-positive={s['nonpositive_count']}"
              + (f" | notes: {'; '.join(notes)}" if notes else ""))
    return pm


# ---------------------------------------------------------------------------------
# Survey-adaptive data contract: borehole / airborne / ground
# ---------------------------------------------------------------------------------
# airborne = the profile runs along the flight line and the late-time window extension is the main mechanism ("toward
# depth" = later times).
SURVEY_MODES = ("borehole", "airborne", "ground")
# profile-axis names of matrix files: a hole word decides borehole first ("along-hole distance" is a borehole axis);
# only an explicit line word makes a line survey
_HOLE_NAME_RE = re.compile(r"depth|hole|well|\bmd\b|深度|孔|井")
_LINE_AXIS_HINTS = ("flight", "line", "easting", "northing", "utm", "测线", "航", "里程")
_LINE_COL_RE = re.compile(r"^(line|flt|flight|flightline|flight_line)[ _\-]?(no|num|number|id|name)?$|^(测线|线号|航线)(号)?$")
_X_COL_NAMES = ("utmx", "utm_x", "easting", "east", "x", "x_m", "xm", "coordx")
_Y_COL_NAMES = ("utmy", "utm_y", "northing", "north", "y", "y_m", "ym", "coordy")
_DIST_COL_NAMES = ("distance", "dist", "along", "along_line", "station_m", "里程", "距离")
_DEPTH_COL_NAMES = ("depth", "depth_m", "depth(m)", "depth (m)", "md", "md_m", "深度", "孔深", "井深", "深度(m)")
# a station column (STATION, CH1..CH31: the Liuyuan borehole table) of a file without coordinate or line columns holds
# the station positions in m
_STATION_COL_NAMES = ("station", "stn", "测点", "测点号", "点号")
# sensor clearance only (a plain 'height' / 高度 column is as often the ground elevation)
_HEIGHT_COL_NAMES = ("ralt", "radar", "radar_alt", "radalt", "laser", "laser_alt", "lidar", "clearance",
                     "terrain_clearance", "tx_height", "rx_height", "txheight", "rxheight", "sensor_height", "bird_height",
                     "height_agl", "agl", "离地高度", "离地高")
_GATE_COL_RE = re.compile(r"^\s*([A-Za-z一-鿿_]+?)[ _\-]?(\d+)\s*$")


@dataclass
class SurveySpec:
    mode: str                 # borehole | airborne | ground
    axis: str                 # "depth" (along the hole) | "distance" (along the line)
    axis_label: str           # profile-axis label of the CSV files and figures
    index_label: str          # 1-based profile-axis index column of the decay-curve files
    anomaly_family: str       # "borehole": near-hole zone / conductor along the hole; "buried": conductor under a line
    sensor_height_m: float    # airborne sensor clearance (m) from the data or --sensor_height; ground 0; borehole nan
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not np.isfinite(d["sensor_height_m"]):
            d["sensor_height_m"] = None          # strict JSON: null where the clearance does not apply
        return d


def _strict_float(tok: Any) -> float:
    try:
        return float(str(tok).strip().strip('"').strip("'"))
    except Exception:
        return float("nan")


def _gate_column_groups(cells: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    """prefix -> [(number, column index)] of the columns named <prefix><number> (zoff1, ch_12, gate 3 ...)."""
    out: Dict[str, List[Tuple[int, int]]] = {}
    for j, c in enumerate(cells):
        m = _GATE_COL_RE.match(str(c))
        if m:
            out.setdefault(m.group(1).lower(), []).append((int(m.group(2)), j))
    return {k: sorted(v) for k, v in out.items() if len(v) >= 3}


def is_survey_line_file(path: str) -> bool:
    """Line files name their columns (geometry + gate columns <prefix><n>); matrix files carry a numeric axis in
    their header row."""
    path = sanitize_path(path)
    if not path or not os.path.isfile(path):
        return False
    enc = _detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    delim = _detect_delimiter(header)
    cells = [c.strip().lstrip("\ufeff") for c in (header.split(delim) if delim != r"\s+" else header.split())]
    if len(cells) < 4:
        return False
    numeric = sum(bool(np.isfinite(_strict_float(c))) for c in cells[1:])
    groups = _gate_column_groups(cells)
    return numeric < 0.5 * (len(cells) - 1) and bool(groups)


def _find_col(cols_lc: Sequence[str], names: Sequence[str]) -> int:
    for nm in names:
        if nm in cols_lc:
            return list(cols_lc).index(nm)
    return -1


def _find_col_re(cols_lc: Sequence[str], rx: "re.Pattern") -> int:
    for j, c in enumerate(cols_lc):
        if rx.match(c.strip()):
            return j
    return -1


def _read_gate_times(path: str) -> np.ndarray:
    """Gate times of a gate-time file: one time per line, or a table (gate number, time ...) whose time column is the one
    named time / centre / 时间 in its header, else its last column.  The times must be positive and strictly increasing
    (a clear error otherwise: a gate-number column read as times would interleave two axes)."""
    enc = _detect_encoding(path)
    rows: List[List[float]] = []
    header: List[str] = []
    with open(path, "r", encoding=enc, errors="replace") as f:
        for line in f:
            s = line.strip().lstrip("\ufeff")
            if not s:
                continue
            toks = [t.strip().strip('"').strip("'") for t in (re.split(r"[,;\t]", s) if re.search(r"[,;\t]", s)
                                                              else s.split())]
            toks = [t for t in toks if t != ""]
            vals = [_strict_float(t) for t in toks]
            if vals and all(np.isfinite(v) for v in vals):
                rows.append(vals)
            elif not rows and not header:
                header = [t.lower() for t in toks]
    if not rows:
        return np.zeros(0)
    ncol = min(len(r) for r in rows)
    j = ncol - 1
    for k, h in enumerate(header[:ncol]):
        if any(w in h for w in ("time", "centre", "center", "时间", "时刻")):
            j = k
            break
    t = np.asarray([r[j] for r in rows], dtype=np.float64)
    if t.size and not (np.all(t > 0) and np.all(np.diff(t) > 0)):
        raise ValueError(f"[DATA] gate-time file {path}: column {j + 1} of {ncol} holds no positive, strictly increasing "
                         f"gate times (first values {np.round(t[:6], 6).tolist()}); one time per line, or a column named "
                         f"time / centre")
    return t


def _find_gate_time_file(data_path: str) -> str:
    """The one gate-time file next to a line file: name with 'gate' and 'time' (or 时间), positive increasing values."""
    folder = os.path.dirname(os.path.abspath(data_path))
    cands = []
    for nm in sorted(os.listdir(folder)):
        low = nm.lower()
        p = os.path.join(folder, nm)
        if "gate" in low and ("time" in low or "时间" in low) and os.path.isfile(p) and p != os.path.abspath(data_path):
            try:
                t = _read_gate_times(p)
            except Exception:
                continue
            if t.size >= 3:
                cands.append(p)
    return cands[0] if len(cands) == 1 else ""


def read_survey_lines(path: str, role: str, gate_time_file: str = "", line_id: str = "auto", component_block: int = 1,
                      gate_window: str = "auto", min_positive_frac: float = 0.99, time_unit: str = "auto",
                      verbose: bool = True) -> ProfileMatrix:
    """Airborne / ground line file -> ProfileMatrix over the along-line distance (m).

    Columns: geometry (line id, UTM x / y or a distance column, sensor height ...) and gate columns <prefix><n> holding
    one or more component blocks (e.g. zoff1..zoff50 = Z gates 1..26 + another component).  The gate times come from
    --gate_time_file or from the one gate-time file in the same folder; their count sets the block length.  One line is
    read (the one with the most stations, or --line_id); the distance is the cumulative station-to-station distance.
    The gate window is the longest run of leading gates whose values are positive at >= min_positive_frac of the
    stations while the median decay still falls (the usable signal window; the later gates are reported and not read),
    or the 1-based range --gate_window a-b."""
    path = sanitize_path(path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"[DATA] {role} file does not exist: {path}")
    enc = _detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        header = f.readline().rstrip("\r\n")
    delim = _detect_delimiter(header)
    df = pd.read_csv(path, sep=delim if delim != r"\s+" else r"\s+", encoding=enc,
                     engine="c" if delim != r"\s+" else "python")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    cols = list(df.columns)
    cols_lc = [c.lower() for c in cols]
    notes: List[str] = []
    groups = _gate_column_groups(cols)
    if not groups:
        raise ValueError(f"[DATA] {role}: no gate columns <prefix><number> in {path}")
    prefix = max(groups, key=lambda k: len(groups[k]))
    gcols = [cols[j] for _, j in groups[prefix]]
    gt_src = sanitize_path(gate_time_file) if gate_time_file else _find_gate_time_file(path)
    t_raw = _read_gate_times(gt_src) if gt_src else np.zeros(0)
    if t_raw.size < 3:
        raise ValueError(f"[DATA] {role}: the gate columns '{prefix}1..' carry no times and no gate-time file was found "
                         f"next to {path}; pass --gate_time_file (one gate time per line)")
    n_g = int(t_raw.size)
    nblk = len(gcols) // n_g
    if nblk < 1:
        raise ValueError(f"[DATA] {role}: {len(gcols)} '{prefix}' columns but {n_g} gate times in {gt_src}")
    b = int(component_block)
    if not 1 <= b <= nblk:
        raise ValueError(f"[DATA] {role}: --component_block {b} outside 1..{nblk}")
    sel_cols = gcols[(b - 1) * n_g: b * n_g]
    if len(gcols) != nblk * n_g:
        notes.append(f"{len(gcols)} '{prefix}' columns = {nblk} block(s) of {n_g} gates + {len(gcols) - nblk * n_g} "
                     f"columns not read")
    # ---- line ----
    jl = _find_col_re(cols_lc, _LINE_COL_RE)
    line_used = ""
    n_lines = 1
    if jl >= 0:
        lc = df.iloc[:, jl]
        num = pd.to_numeric(lc, errors="coerce")
        if bool(num.notna().all()) and bool(np.all(np.isclose(num, np.round(num)))):
            lines = num.round().astype(np.int64).astype(str)           # 12 and 12.0 name the same line
        else:
            lines = lc.astype(str).str.strip()
        counts = lines.value_counts()
        n_lines = int(counts.size)
        lid = str(line_id or "auto").strip()
        if lid.lower() == "auto":
            lid = str(counts.index[0])
        if lid not in counts.index:
            raise ValueError(f"[DATA] {role}: line {lid!r} not in the file (lines: {list(counts.index)[:20]})")
        if n_lines > 1:
            notes.append(f"{n_lines} lines in the file; line {lid} ({int(counts[lid])} stations) read")
        df = df[lines == lid].reset_index(drop=True)
        line_used = lid
    # ---- profile coordinate: a depth column (borehole table), else the along-line distance ----
    def _num(j: int) -> np.ndarray:
        return pd.to_numeric(df.iloc[:, j], errors="coerce").to_numpy(np.float64)

    def _varies(v: np.ndarray) -> bool:
        v = v[np.isfinite(v)]
        return v.size >= 2 and float(np.ptp(v)) > 0.0

    jz = _find_col(cols_lc, _DEPTH_COL_NAMES)
    jx, jy, jd = _find_col(cols_lc, _X_COL_NAMES), _find_col(cols_lc, _Y_COL_NAMES), _find_col(cols_lc, _DIST_COL_NAMES)
    xy_ok = jx >= 0 and jy >= 0 and (_varies(_num(jx)) or _varies(_num(jy)))
    js = _find_col(cols_lc, _STATION_COL_NAMES)
    sv = _num(js) if js >= 0 else np.zeros(0)
    svf = np.unique(sv[np.isfinite(sv)])
    st_pos = (js >= 0 and jl < 0 and _varies(sv)
              and not (np.all(np.abs(svf - np.round(svf)) < 1e-9) and np.all(np.abs(np.diff(svf) - 1.0) < 1e-9)))
    axis_kind = "distance"
    if jz >= 0 and _varies(_num(jz)):
        dist = _num(jz)
        ok = np.isfinite(dist)
        df, dist = df[ok].reset_index(drop=True), dist[ok]
        coord_src, axis_name, axis_kind = f"depth column '{cols[jz]}'", "depth (m)", "depth"
    elif xy_ok:
        x, y = _num(jx), _num(jy)
        ok = np.isfinite(x) & np.isfinite(y)
        df, x, y = df[ok].reset_index(drop=True), x[ok], y[ok]
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        coord_src, axis_name = f"cumulative station distance of ({cols[jx]}, {cols[jy]})", "distance (m)"
    elif jd >= 0 and _varies(_num(jd)):
        dist = _num(jd)
        ok = np.isfinite(dist)
        df, dist = df[ok].reset_index(drop=True), dist[ok]
        dist = dist - float(dist.min())
        coord_src, axis_name = f"column '{cols[jd]}'", "distance (m)"
    elif st_pos:
        # the station column holds the positions (Liuyuan: 10-256 m, 5 / 3 / 1 m spacing): they are the profile axis, as
        # measured
        ok = np.isfinite(sv)
        df, dist = df[ok].reset_index(drop=True), sv[ok]
        coord_src, axis_name, axis_kind = f"station column '{cols[js]}' (positions, m)", "station (m)", "station"
    else:
        dist = np.arange(len(df), dtype=np.float64)
        coord_src, axis_name = "station order (no varying coordinate column; 1 unit = 1 station)", "station"
        notes.append("no varying coordinate column: the profile axis is the station number")
    if axis_kind == "distance" and dist.size >= 3 and jl < 0:
        # no line column: a jump far beyond the station spacing joins two lines; the longest stretch is read
        step = np.abs(np.diff(dist))
        med = float(np.median(step[step > 0])) if np.any(step > 0) else 0.0
        cut = np.nonzero(step > max(20.0 * med, 0.25 * float(dist[-1] - dist[0])))[0] if med > 0 else np.zeros(0, int)
        if cut.size:
            seg = np.split(np.arange(dist.size), cut + 1)
            best = max(seg, key=len)
            notes.append(f"no line column and {cut.size} jump(s) > 20 x the station spacing: {len(seg)} stretches, the "
                         f"longest ({best.size} stations) read")
            df, dist = df.iloc[best].reset_index(drop=True), dist[best] - float(dist[best][0])
    jh = _find_col(cols_lc, _HEIGHT_COL_NAMES)
    h_med = float(np.nanmedian(pd.to_numeric(df.iloc[:, jh], errors="coerce"))) if jh >= 0 else float("nan")
    V = df[sel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    fac, unit, unit_reason = _time_unit_to_ms(t_raw, time_unit, "")
    t_ms = t_raw * fac
    # ---- usable gate window ----
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a gate without any positive value: median NaN
        fin = np.isfinite(V)
        # share of positive values among the recorded (finite) samples: a missing sample is not a sign change
        pos = np.sum(fin & (V > 0), axis=0) / np.maximum(np.sum(fin, axis=0), 1)
        LV = np.where(V > 0, np.log10(np.where(V > 0, V, 1.0)), np.nan)
        Lmed = np.nanmedian(LV, axis=0)
        # the decay of the typical station: median over the stations of the change from the previous gate.  The change
        # of the per-gate median is not the median change: the Liuyuan median value at gate 31 (16 ms) lies 0.002
        # decades above gate 30 while 50 of 57 stations still fall (median change -0.011), and gate 31 was dropped
        dmed = np.r_[np.nan, np.nanmedian(np.diff(LV, axis=1), axis=0)] if n_g > 1 else np.full(n_g, np.nan)
    gw = str(gate_window or "auto").strip().lower()
    if gw == "auto":
        # the longest run of consecutive usable gates (positive at >= min_positive_frac, median decay still falling);
        # leading gates spoilt by the transmitter ramp / primary field are skipped, not the whole window
        good = (pos >= float(min_positive_frac)) & np.isfinite(Lmed)
        g0, g1, a_ = 0, 0, -1
        for g in range(n_g):
            if not good[g]:
                a_ = -1
                continue
            if a_ < 0 or not dmed[g] < 0:
                a_ = g
            if g + 1 - a_ > g1 - g0:
                g0, g1 = a_, g + 1
        rule = (f"usable-gate rule (longest run of gates positive at >= {float(min_positive_frac):.0%} of the stations "
                f"with a falling median decay)")
    else:
        a_, _, b_ = gw.partition("-")
        try:
            g0, g1 = int(a_) - 1, int(b_ or a_)
        except ValueError:
            raise ValueError(f"[DATA] {role}: --gate_window expects auto or a-b (1-based gates), got {gate_window!r}")
        if not 0 <= g0 < g1 <= n_g:
            raise ValueError(f"[DATA] {role}: --gate_window {gate_window} outside the gates 1..{n_g} (a <= b)")
        rule = f"--gate_window {gate_window}"
    if g1 - g0 < 3:
        raise ValueError(f"[DATA] {role}: only {g1 - g0} usable gates ({rule}); positive share per gate "
                         f"{np.round(pos, 3).tolist()} (choose the gates with --gate_window a-b)")
    dropped = [{"gate": g + 1, "t_ms": float(t_ms[g]), "positive_share": float(pos[g])}
               for g in range(n_g) if not g0 <= g < g1]
    V, t_ms = V[:, g0:g1], t_ms[g0:g1]
    fin_row = np.isfinite(V).mean(axis=1) >= 0.5
    if not np.all(fin_row):
        notes.append(f"dropped {int((~fin_row).sum())} stations with <50% finite values")
        V, dist = V[fin_row], dist[fin_row]
    # ---- order / duplicates along the line ----
    di = np.argsort(dist, kind="mergesort")
    dist, V = dist[di], V[di]
    if np.any(np.diff(dist) <= 0):
        ud, inv = np.unique(dist, return_inverse=True)
        V = _average_rows(V, inv, ud.size)
        notes.append(f"{dist.size - ud.size} stations at repeated positions averaged")
        dist = ud
    if dist.size < 4:
        raise ValueError(f"[DATA] {role}: only {dist.size} station(s) left on line {line_used or '-'} after reading "
                         f"({coord_src}); a profile needs >= 4 stations")
    V, n_fill = _fill_nonfinite_rows(t_ms, V)
    if n_fill:
        notes.append(f"filled {n_fill} non-finite samples by log-time interpolation")
    meta = {"format": "survey_lines", "axis_kind": axis_kind, "gate_prefix": prefix,
            "gate_columns": [sel_cols[g0], sel_cols[g1 - 1]],
            "component_block": b, "n_component_blocks": nblk, "gate_time_file": gt_src, "line": line_used,
            "n_lines_in_file": n_lines, "coordinate": coord_src, "gate_window_rule": rule,
            "gates_read": [g0 + 1, g1], "gates_dropped": dropped,
            "height_column": cols[jh] if jh >= 0 else "", "sensor_height_m": h_med if np.isfinite(h_med) else None}
    pm = ProfileMatrix(path=path, role=role, depth=dist.astype(np.float64), time_ms=t_ms.astype(np.float64),
                       values=V.astype(np.float64), orientation="rows=stations (line file)", time_unit=unit,
                       time_unit_reason=f"{unit_reason} (gate-time file)", orientation_reason="line file columns",
                       axis_name=axis_name, encoding=enc, sha16=file_sha256(path)[:16], notes=notes, meta=meta)
    if verbose:
        s = pm.summary()
        print("=" * 88)
        print(f"[DATA-READ] {role}: {path} (line file)")
        print(f"[DATA-READ]   line {line_used or '-'} of {n_lines} | coordinate = {coord_src} | gate columns "
              f"{sel_cols[g0]}..{sel_cols[g1 - 1]} (block {b}/{nblk}) | gate times {gt_src} ({unit_reason})")
        print(f"[DATA-READ]   gates {g0 + 1}..{g1} of {n_g} read ({rule}): {t_ms[0]:.6g} .. {t_ms[-1]:.6g} ms"
              + (f" | not read: " + ", ".join(f"g{d['gate']} {d['t_ms']:.4g} ms pos {d['positive_share']:.0%}"
                                              for d in dropped[:8]) + (" ..." if len(dropped) > 8 else "") if dropped else ""))
        print(f"[DATA-READ]   {s['n_depth']} stations over {s['depth_min_m']:.6g} .. {s['depth_max_m']:.6g} "
              f"{'m' if axis_name != 'station' else 'stations'} (spacing median {s['depth_spacing_m']['median']:.4g})"
              + (f" | sensor height ({cols[jh]}) median {h_med:.1f} m" if jh >= 0 else "")
              + (f" | notes: {'; '.join(notes)}" if notes else ""))
    return pm


def stack_stations(pm: ProfileMatrix, max_stations: int, verbose: bool = True) -> ProfileMatrix:
    """Stack consecutive stations of a long profile (linear mean, as a receiver stacks) so that at most max_stations
    remain; a stacked station sits at the mean coordinate of its members.  0 = off."""
    D = int(pm.depth.size)
    if 0 < int(max_stations) < 4:
        raise ValueError(f"--max_stations {max_stations}: a profile needs >= 4 stations (0 = no stacking)")
    if int(max_stations) <= 0 or D <= int(max_stations):
        return pm
    k = int(math.ceil(D / int(max_stations)))
    nb = int(math.ceil(D / k))
    idx = np.minimum(np.arange(D) // k, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    d = np.bincount(idx, weights=pm.depth, minlength=nb) / cnt
    V = np.zeros((nb, pm.values.shape[1]))
    np.add.at(V, idx, pm.values)
    V /= cnt[:, None]
    pm2 = copy.copy(pm)
    pm2.depth, pm2.values = d.astype(np.float64), V
    pm2.notes = list(pm.notes) + [f"stacked {k} consecutive stations: {D} -> {nb} stations (--max_stations {max_stations})"]
    pm2.meta = dict(pm.meta)
    pm2.meta["stacking"] = {"stations_in": D, "stations_per_stack": k, "stations_out": nb}
    if verbose:
        dd = np.diff(d)
        print(f"[DATA-READ]   stacked {k} consecutive stations: {D} -> {nb} stations (spacing median "
              f"{float(np.median(dd)):.4g}; --max_stations {max_stations})")
    return pm2


def read_measured_file(path: str, role: str, data_cfg: "DataConfig", time_unit: str, verbose: bool = True) -> ProfileMatrix:
    """Matrix file (header row / column) or airborne / ground line file, then station stacking of long profiles.
    --max_stations -1 (auto) stacks line files to 256 stations and leaves matrix files as read; an explicit value
    applies to both, 0 = off."""
    if is_survey_line_file(path):
        pm = read_survey_lines(path, role, data_cfg.gate_time_file, data_cfg.line_id, data_cfg.component_block,
                               data_cfg.gate_window, data_cfg.min_positive_frac, time_unit, verbose=verbose)
    else:
        pm = read_profile_matrix(path, role, data_cfg.read_by, time_unit, verbose=verbose)
        pm.meta.setdefault("format", "matrix")
    ms = int(data_cfg.max_stations)
    if ms < 0:
        ms = 256 if pm.meta.get("format") == "survey_lines" else 0
    return stack_stations(pm, ms, verbose=verbose)


def detect_survey_mode(pm: ProfileMatrix) -> Tuple[str, str]:
    """Survey mode from the measured file: line files with a depth column are borehole tables, line files with a
    sensor-clearance column airborne, other line files ground.  Matrix files are borehole unless their profile-axis
    name is an explicit line coordinate (flight / line / easting / northing / UTM / 测线 / 航) without a hole word;
    ambiguous names (distance, offset, position, x) stay borehole - pass --survey_mode for such a line. A table on a
    station column with positions (STATION, CH1..) is a borehole too, as a matrix whose profile axis is named
    station: --survey_mode ground / airborne for a line."""
    meta = pm.meta or {}
    if meta.get("format") == "survey_lines":
        if meta.get("axis_kind") == "depth":
            return "borehole", f"line-style table whose profile axis is the depth column ({meta.get('coordinate', '')})"
        if meta.get("axis_kind") == "station" and not meta.get("height_column"):
            return "borehole", (f"station table, profile axis = the station positions ({meta.get('coordinate', '')}): "
                                f"borehole, as a matrix on a station axis (--survey_mode ground / airborne for a line)")
        if meta.get("height_column"):
            return "airborne", f"line file with a sensor-clearance column '{meta['height_column']}'"
        return "ground", "line file without a sensor-clearance column"
    nm = (pm.axis_name or "").strip().lower()
    if _HOLE_NAME_RE.search(nm):
        return "borehole", f"profile-axis name '{pm.axis_name}' names a depth / hole"
    if any(h in nm for h in _LINE_AXIS_HINTS):
        air = any(h in nm for h in ("flight", "航"))
        return ("airborne" if air else "ground"), f"profile-axis name '{pm.axis_name}' is a line coordinate"
    return "borehole", (f"matrix file, profile-axis name '{pm.axis_name}' names no line coordinate -> borehole (the v15 "
                        f"default; --survey_mode airborne / ground for a line)")


def build_survey_spec(mode: str, pm: ProfileMatrix, sensor_height: str = "auto", reason: str = "") -> SurveySpec:
    mode = str(mode or "borehole").lower()
    if mode not in SURVEY_MODES:
        mode = "borehole"
    if mode == "borehole":
        return SurveySpec("borehole", "depth", "depth (m)", "depth_index", "borehole", float("nan"), reason)
    s = str(sensor_height or "auto").strip().lower()
    if s != "auto":
        h = _strict_float(s)
        if not np.isfinite(h) or h < 0:
            raise ValueError(f"--sensor_height expects auto or a clearance in metres (a number >= 0), got {sensor_height!r}")
    elif mode == "airborne":
        hm = (pm.meta or {}).get("sensor_height_m")
        h = float(hm) if hm is not None else float("nan")
    else:
        h = 0.0
    label = "station" if (pm.axis_name or "").strip().lower() == "station" else "distance (m)"
    return SurveySpec(mode, "distance", label, "position_index", "buried", h, reason)
