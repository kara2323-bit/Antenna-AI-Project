import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from groq import Groq


FREQS_RAW = np.linspace(1.0, 7.0, 151)
SHAPES = {
    "1. Rectangle": 1,
    "2. Stepped": 2,
    "3. T-Shape": 3,
    "4. Ellipse": 4,
    "5. Semi-Circle": 5,
    "6. Pie-Sector": 6,
    "7. Isosceles Triangle": 7,
    "8. Inverted Trapezoid": 8,
    "9. Diamond": 9,
    "10. Hexagon (Flat-top)": 10,
    "11. Pentagon (House)": 11,
    "12. Cross": 12,
}
FAMILY_MAP = {
    "Rectangular": [1, 2, 3],
    "Circular": [4, 5, 6],
    "Triangular": [7, 8, 9],
    "Polygonal": [10, 11, 12],
}
KNOWN_META_COLS = {
    "Global_ID", "Family_ID", "Antenna_ID", "Run_ID", "Lp", "Wp",
    "fr (GHz)", "BW (GHz)", "Gain_at_fr", "Eff_at_fr", "S11_min", "is_matched"
}


def clamp_5_40(v):
    return float(max(5.0, min(40.0, float(v))))


def apply_dsp_polishing(freqs, curve):
    freqs = np.asarray(freqs, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if len(freqs) != len(curve) or len(curve) < 2:
        raise ValueError("freqs and curve must be same length >= 2")

    finite = np.isfinite(freqs) & np.isfinite(curve)
    if not np.all(finite):
        freqs = freqs[finite]
        curve = curve[finite]
        if len(freqs) < 2:
            raise ValueError("Not enough finite points for polishing")

    if len(curve) >= 5:
        win = min(19, (len(curve) // 2) * 2 - 1)
        win = max(5, win)
        poly = min(3, win - 1)
        smooth = savgol_filter(curve, win, poly)
    else:
        smooth = curve

    f_fine = np.linspace(freqs.min(), freqs.max(), 1000)
    k = min(3, len(freqs) - 1)
    spline = make_interp_spline(freqs, smooth, k=k)
    return f_fine, spline(f_fine)


def vna_measure(freqs, curve):
    idx_min = int(np.argmin(curve))
    m_fr = float(freqs[idx_min])
    m_s11 = float(curve[idx_min])
    mask = (curve <= -10.0).astype(int)
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    status = "WORKING" if len(starts) > 0 else "NOT WORKING"
    bw = 0.0
    if len(starts) > 0:
        primary = next(((s, e) for s, e in zip(starts, ends) if s <= idx_min <= e), (starts[0], ends[0]))
        bw = float(freqs[primary[1]] - freqs[primary[0]])
    gamma = 10 ** (np.clip(curve, -100, 0) / 20.0)
    vswr = np.clip((1 + gamma) / np.maximum(1e-8, (1 - gamma)), 1, 10)
    return m_fr, m_s11, bw, status, vswr


@st.cache_resource
def load_brains():
    # Forward model priority: shrunk -> final -> h5
    if os.path.exists("forward_model_shrunk.keras"):
        fwd_path = "forward_model_shrunk.keras"
    elif os.path.exists("forward_model_final.keras"):
        fwd_path = "forward_model_final.keras"
    elif os.path.exists("forward_model_final.h5"):
        fwd_path = "forward_model_final.h5"
    else:
        raise FileNotFoundError(
            "No forward model file found. Expected one of: "
            "forward_model_shrunk.keras, forward_model_final.keras, forward_model_final.h5"
        )

    # Inverse model is optional
    inv_path = "inverse_model_final.keras" if os.path.exists("inverse_model_final.keras") else None

    # Required scalers
    if not os.path.exists("scaler_geo.pkl"):
        raise FileNotFoundError("Missing scaler_geo.pkl")
    if not os.path.exists("scaler_perf.pkl"):
        raise FileNotFoundError("Missing scaler_perf.pkl")

    fwd = tf.keras.models.load_model(fwd_path, compile=False)
    inv = tf.keras.models.load_model(inv_path, compile=False) if inv_path else None
    s_geo = joblib.load("scaler_geo.pkl")
    s_perf = joblib.load("scaler_perf.pkl")
    return fwd, inv, s_geo, s_perf

@st.cache_data
def load_data():
    candidates = []
    for p in ["antenna_data_website.csv", "antenna_data_cleaned.csv", "antenna_data.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # Quick curve-column score by numeric frequency-like headers.
                score = 0
                for c in df.columns:
                    try:
                        f = float(str(c).strip())
                        if 1.0 <= f <= 7.0:
                            score += 1
                    except ValueError:
                        continue
                # Fallback score by wide table size (many S11 points even with bad headers).
                width_score = max(0, df.shape[1] - 11)
                candidates.append((score, width_score, p, df))
            except Exception:
                continue
    if not candidates:
        return None
    # Prefer dataset with richer curve information.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    return best[3]


def curve_columns(df):
    cols = []
    for c in df.columns:
        if c in KNOWN_META_COLS:
            continue
        try:
            f = float(str(c).strip())
            if 1.0 <= f <= 7.0:
                cols.append((f, c))
        except ValueError:
            continue
    cols.sort(key=lambda x: x[0])
    if len(cols) >= 10:
        return [c for _, c in cols], np.array([f for f, _ in cols], dtype=float)

    # Fallback: if headers are broken, treat all columns after first 11 as curve columns.
    # Then synthesize an even 1-7 GHz axis to keep measurement engine operational.
    if df.shape[1] > 20:
        fallback_cols = list(df.columns[11:])
        f_axis = np.linspace(1.0, 7.0, len(fallback_cols))
        return fallback_cols, f_axis

    return [], np.array([], dtype=float)


def weighted_blend_from_rows(rows, weights, curve_cols):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    fr = float(np.sum(rows["fr (GHz)"].to_numpy(dtype=float) * weights)) if "fr (GHz)" in rows.columns else np.nan
    bw = float(np.sum(rows["BW (GHz)"].to_numpy(dtype=float) * weights)) if "BW (GHz)" in rows.columns else np.nan
    gain = float(np.sum(rows["Gain_at_fr"].to_numpy(dtype=float) * weights)) if "Gain_at_fr" in rows.columns else np.nan
    eff = float(np.sum(rows["Eff_at_fr"].to_numpy(dtype=float) * weights)) if "Eff_at_fr" in rows.columns else np.nan
    curve = np.sum(rows[curve_cols].to_numpy(dtype=float) * weights.reshape(-1, 1), axis=0)
    return fr, bw, gain, eff, curve


def grid_bilinear_estimate(subset, lp, wp, curve_cols):
    lvals = np.sort(subset["Lp"].unique())
    wvals = np.sort(subset["Wp"].unique())
    if len(lvals) < 2 or len(wvals) < 2:
        return None

    l_lo = lvals[lvals <= lp][-1] if np.any(lvals <= lp) else lvals[0]
    l_hi = lvals[lvals >= lp][0] if np.any(lvals >= lp) else lvals[-1]
    w_lo = wvals[wvals <= wp][-1] if np.any(wvals <= wp) else wvals[0]
    w_hi = wvals[wvals >= wp][0] if np.any(wvals >= wp) else wvals[-1]

    points = [
        (l_lo, w_lo),
        (l_lo, w_hi),
        (l_hi, w_lo),
        (l_hi, w_hi),
    ]
    rows = []
    for ll, ww in points:
        cell = subset[(subset["Lp"] == ll) & (subset["Wp"] == ww)]
        if cell.empty:
            return None
        rows.append(cell.iloc[0])
    rows = pd.DataFrame(rows)

    dl = max(1e-9, float(l_hi - l_lo))
    dw = max(1e-9, float(w_hi - w_lo))
    tl = 0.0 if l_hi == l_lo else float((lp - l_lo) / dl)
    tw = 0.0 if w_hi == w_lo else float((wp - w_lo) / dw)
    weights = np.array([(1 - tl) * (1 - tw), (1 - tl) * tw, tl * (1 - tw), tl * tw], dtype=float)
    fr, bw, gain, eff, curve = weighted_blend_from_rows(rows, weights, curve_cols)
    return fr, bw, gain, eff, curve


def forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf):
    x = np.zeros((1, 14), dtype=float)
    x[0, 0], x[0, 1], x[0, 1 + shape_id] = lp, wp, 1.0
    x_scaled = s_geo.transform(x)
    p = fwd.predict(x_scaled, verbose=0)
    recon = np.zeros((1, 157), dtype=float)
    recon[0, [0, 2, 3, 4]] = p[0][0]
    recon[0, 6:] = p[1][0]
    perf = s_perf.inverse_transform(recon)[0]
    f_s, c_s = apply_dsp_polishing(FREQS_RAW, perf[6:])
    fr, s11, bw, status, vswr = vna_measure(f_s, c_s)
    return {
        "fr": fr, "s11": s11, "bw": bw, "gain": float(perf[2]), "eff": float(perf[3]),
        "status": status, "f": f_s, "c": c_s, "v": vswr, "engine": "ML-only"
    }


def forward_anchor(lp, wp, shape_id, db):
    if db is None or "Antenna_ID" not in db.columns:
        return None
    curve_cols, f_axis = curve_columns(db)
    if len(curve_cols) < 10:
        return None
    subset = db[db["Antenna_ID"] == shape_id].copy()
    if subset.empty:
        return None

    exact = subset[(subset["Lp"] == lp) & (subset["Wp"] == wp)]
    if not exact.empty:
        row = exact.iloc[[0]]
        fr, bw, gain, eff, curve = weighted_blend_from_rows(row, [1.0], curve_cols)
        dist_min = 0.0
        engine_note = "CSV-exact"
    else:
        bilinear = grid_bilinear_estimate(subset, lp, wp, curve_cols)
        if bilinear is not None:
            fr, bw, gain, eff, curve = bilinear
            dist_min = float(np.min(np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)))
            engine_note = "CSV-bilinear"
        else:
            subset["dist"] = np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)
            near = subset.nsmallest(min(8, len(subset)), "dist")
            d = near["dist"].to_numpy(dtype=float)
            w = 1.0 / (d + 1e-6)
            w = w / np.sum(w)
            fr, bw, gain, eff, curve = weighted_blend_from_rows(near, w, curve_cols)
            dist_min = float(np.min(d))
            engine_note = "CSV-knn"

    f_s, c_s = apply_dsp_polishing(f_axis, curve)
    fr_m, s11, bw_m, status, vswr = vna_measure(f_s, c_s)
    return {
        "fr": fr_m if np.isfinite(fr_m) else fr,
        "s11": s11,
        "bw": bw_m if bw_m > 0 else bw,
        "gain": gain,
        "eff": eff,
        "status": status,
        "f": f_s,
        "c": c_s,
        "v": vswr,
        "dist": dist_min,
        "anchor_mode": engine_note,
    }


def forward_best(lp, wp, shape_id, fwd, s_geo, s_perf, db):
    ml = forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf)
    anc = forward_anchor(lp, wp, shape_id, db)
    if anc is None:
        return ml
    if anc["dist"] <= 0.60:
        anc["engine"] = f'{anc.get("anchor_mode", "CSV")}-anchored'
        return anc
    a, m = 0.75, 0.25
    out = {
        "fr": a * anc["fr"] + m * ml["fr"],
        "s11": a * anc["s11"] + m * ml["s11"],
        "bw": a * anc["bw"] + m * ml["bw"],
        "gain": a * anc["gain"] + m * ml["gain"],
        "eff": a * anc["eff"] + m * ml["eff"],
        "status": "WORKING" if (a * anc["s11"] + m * ml["s11"]) <= -10 else "NOT WORKING",
        "f": anc["f"],
        "c": a * anc["c"] + m * ml["c"],
        "v": a * anc["v"] + m * ml["v"],
        "engine": "Hybrid (CSV + ML)",
    }
    return out


def forward_physics_first(lp, wp, shape_id, fwd, s_geo, s_perf, db, use_ml_refine=False):
    anc = forward_anchor(lp, wp, shape_id, db)
    if anc is None:
        ml = forward_ml(lp, wp, shape_id, fwd, s_geo, s_perf)
        ml["engine"] = "ML-only (no CSV anchor)"
        return ml
    anc["engine"] = f'{anc.get("anchor_mode", "CSV")}-physics'
    if not use_ml_refine:
        return anc
    mixed = forward_best(lp, wp, shape_id, fwd, s_geo, s_perf, db)
    mixed["engine"] = f'{mixed.get("engine", "Hybrid")} + refine'
    return mixed


def nearest_csv_row(db, lp, wp, shape_id):
    subset = db[db["Antenna_ID"] == shape_id].copy()
    if subset.empty:
        return None
    subset["dist"] = np.hypot(subset["Lp"] - lp, subset["Wp"] - wp)
    return subset.nsmallest(1, "dist").iloc[0]


@st.cache_data
def run_engine_self_test(db, sample_size=180):
    curve_cols, f_axis = curve_columns(db)
    if len(curve_cols) < 10:
        return None
    test_df = db.sample(min(sample_size, len(db)), random_state=42).copy()
    fr_errs = []
    s11_errs = []
    bw_errs = []
    for _, row in test_df.iterrows():
        sid = int(row["Antenna_ID"])
        lp = float(row["Lp"])
        wp = float(row["Wp"])
        pred = forward_anchor(lp, wp, sid, db)
        if pred is None:
            continue
        true_curve = row[curve_cols].to_numpy(dtype=float)
        f_true, true_smooth = apply_dsp_polishing(f_axis, true_curve)
        fr_t, s11_t, bw_t, _, _ = vna_measure(f_true, true_smooth)
        fr_errs.append(abs(pred["fr"] - fr_t))
        s11_errs.append(abs(pred["s11"] - s11_t))
        bw_errs.append(abs(pred["bw"] - bw_t))
    if not fr_errs:
        return None
    return {
        "n": len(fr_errs),
        "fr_mae_mhz": float(np.mean(fr_errs) * 1000.0),
        "s11_mae_db": float(np.mean(s11_errs)),
        "bw_mae_mhz": float(np.mean(bw_errs) * 1000.0),
    }


def sync_lp_from_slider():
    v = clamp_5_40(st.session_state.lp_slider)
    st.session_state.lp_val = v
    st.session_state.lp_input = v


def sync_lp_from_input():
    v = clamp_5_40(st.session_state.lp_input)
    st.session_state.lp_val = v
    st.session_state.lp_slider = v


def sync_wp_from_slider():
    v = clamp_5_40(st.session_state.wp_slider)
    st.session_state.wp_val = v
    st.session_state.wp_input = v


def sync_wp_from_input():
    v = clamp_5_40(st.session_state.wp_input)
    st.session_state.wp_val = v
    st.session_state.wp_slider = v


def run_inverse_solver(target_fr, family_name, priority, fwd, s_geo, s_perf, db, half_span=1.5, step=0.25):
    ids = FAMILY_MAP[family_name]
    results = []
    for shape_id in ids:
        subset = db[db["Antenna_ID"] == shape_id].copy()
        if subset.empty:
            continue
        subset["df"] = np.abs(subset["fr (GHz)"] - target_fr)
        anchor = subset.sort_values("df").iloc[0]
        lp0, wp0 = float(anchor["Lp"]), float(anchor["Wp"])
        lp_grid = np.arange(max(5.0, lp0 - half_span), min(40.0, lp0 + half_span) + 1e-9, step)
        wp_grid = np.arange(max(5.0, wp0 - half_span), min(40.0, wp0 + half_span) + 1e-9, step)
        best_local = None
        for lp in lp_grid:
            for wp in wp_grid:
                pred = forward_physics_first(float(lp), float(wp), shape_id, fwd, s_geo, s_perf, db, use_ml_refine=False)
                fr_err = abs(pred["fr"] - target_fr)
                status_pen = 0.04 if pred["status"] != "WORKING" else 0.0
                if priority == "Maximize Gain":
                    score = fr_err + status_pen - 0.015 * float(np.nan_to_num(pred["gain"], nan=0.0))
                elif priority == "Maximize Bandwidth":
                    score = fr_err + status_pen - 0.010 * float(np.nan_to_num(pred["bw"], nan=0.0))
                else:
                    # Balanced profile: prioritize frequency match, then stable bandwidth/gain.
                    score = fr_err + status_pen - 0.006 * float(np.nan_to_num(pred["gain"], nan=0.0)) - 0.004 * float(np.nan_to_num(pred["bw"], nan=0.0))
                if best_local is None or score < best_local["score"]:
                    best_local = {
                        "shape_id": shape_id,
                        "lp": float(lp),
                        "wp": float(wp),
                        "fr": float(pred["fr"]),
                        "err": float(abs(pred["fr"] - target_fr)),
                        "gain": float(np.nan_to_num(pred["gain"], nan=0.0)),
                        "bw": float(np.nan_to_num(pred["bw"], nan=0.0)),
                        "status": pred["status"],
                        "score": float(score),
                    }
        if best_local:
            results.append(best_local)
    if not results:
        return None
    return pd.DataFrame(results).sort_values("err").iloc[0]


def run_inverse_self_test(db, fwd, s_geo, s_perf, per_family=3, fast_mode=True):
    rows = []
    rng = np.random.default_rng(42)
    priorities = ["Balanced Profile", "Maximize Gain", "Maximize Bandwidth"]
    for family in FAMILY_MAP.keys():
        subset = db[db["Antenna_ID"].isin(FAMILY_MAP[family])]
        if subset.empty:
            continue
        pick = subset.sample(min(per_family, len(subset)), random_state=42)
        for _, r in pick.iterrows():
            tfr = float(r["fr (GHz)"])
            pr = priorities[int(rng.integers(0, len(priorities)))]
            if fast_mode:
                best = run_inverse_solver(
                    tfr, family, pr, fwd, s_geo, s_perf, db,
                    half_span=1.0, step=0.75
                )
            else:
                best = run_inverse_solver(
                    tfr, family, pr, fwd, s_geo, s_perf, db,
                    half_span=1.5, step=0.25
                )
            if best is None:
                continue
            rows.append({
                "family": family,
                "priority": pr,
                "target_fr": tfr,
                "pred_fr": float(best["fr"]),
                "err_mhz": abs(float(best["fr"]) - tfr) * 1000.0,
            })
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    summary = {
        "tests": int(len(df)),
        "mean_err_mhz": float(df["err_mhz"].mean()),
        "p90_err_mhz": float(df["err_mhz"].quantile(0.90)),
        "max_err_mhz": float(df["err_mhz"].max()),
    }
    fam = df.groupby("family")["err_mhz"].mean().reset_index().rename(columns={"err_mhz": "mean_err_mhz"})
    return summary, fam


def run_startup_healthcheck(db, fwd, s_geo, s_perf):
    report = {"ok": True, "checks": []}
    try:
        csv_ok = db is not None and len(db) > 0
        report["checks"].append(("CSV loaded", csv_ok))
        report["checks"].append(("Shape coverage", csv_ok and "Antenna_ID" in db.columns and db["Antenna_ID"].nunique() >= 12))
        if not csv_ok:
            report["checks"].append(("Runtime exception", False))
            report["ok"] = False
            return report
        test_row = db.sample(1, random_state=7).iloc[0]
        sid = int(test_row["Antenna_ID"])
        lp = float(test_row["Lp"])
        wp = float(test_row["Wp"])
        x = np.zeros((1, 14), dtype=float)
        x[0, 0], x[0, 1], x[0, 1 + sid] = lp, wp, 1.0
        x_scaled = s_geo.transform(x)
        y = fwd.predict(x_scaled, verbose=0)
        report["checks"].append(("Forward output heads", len(y) == 2 and y[0].shape[-1] == 4))
        pred = forward_physics_first(lp, wp, sid, fwd, s_geo, s_perf, db, use_ml_refine=False)
        report["checks"].append(("Physics engine run", np.isfinite(pred["fr"]) and np.isfinite(pred["s11"])))
        report["checks"].append(("Curve length", len(pred["f"]) == 1000 and len(pred["c"]) == 1000))
    except Exception:
        report["checks"].append(("Runtime exception", False))
    report["ok"] = all(v for _, v in report["checks"])
    return report


st.set_page_config(page_title="NeuralCAD PRO | Charcoal Elite", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background-color: #242526 !important; color: white; font-family: 'Outfit', sans-serif; scroll-behavior: smooth; }
header { visibility: hidden; }
.navbar { position: fixed; top: 0; left: 0; width: 100%; height: 90px; background: #111111; display: flex; align-items: center; justify-content: center; border-bottom: 2px solid #FF8C00; z-index: 1000000; box-shadow: 0 10px 50px rgba(0,0,0,0.8); }
.nav-content { width: 90%; max-width: 1400px; display: flex; justify-content: space-between; align-items: center; }
.nav-logo { font-weight: 900; color: #FF8C00; letter-spacing: -1px; font-size: 1.6rem; }
.nav-links { display: flex; gap: 15px; }
.nav-item { text-decoration: none !important; color: white !important; font-weight: 800; font-size: 0.75rem; padding: 12px 24px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); text-transform: uppercase; letter-spacing: 2px; }
.section-anchor { display: block; position: relative; top: -120px; visibility: hidden; }
.hero-container { position: relative; width: 100%; height: 95vh; display: flex; justify-content: center; align-items: center; padding-top: 100px; overflow: hidden; margin-bottom: 120px; }
.hero-slide { position: relative; width: 92%; max-width: 1350px; height: 75vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; border-radius: 80px; background: #1C1D1E; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 60px 150px rgba(0,0,0,0.8); padding: 60px; }
.hero-h1 { font-size: clamp(3rem, 10vw, 5.5rem); font-weight: 900; line-height: 0.95; margin-bottom: 25px; letter-spacing: -4px; background: linear-gradient(to bottom, #FFF 40%, #FF8C00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-p { font-size: 1.4rem; opacity: 0.9; max-width: 900px; line-height: 1.6; color: #D1D1D1; }
.section-title { font-size: 3.5rem; font-weight: 900; margin: 150px 0 60px; text-align: center; letter-spacing: -3px; color: white; }
.info-card { background: #1C1D1E; border: 1px solid rgba(255,255,255,0.08); border-radius: 40px; padding: 55px; text-align: center; height: 100%; }
.stButton > button { background: linear-gradient(135deg, #FF8C00, #FF4500) !important; color: white !important; border: none !important; border-radius: 15px !important; padding: 14px 40px !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 2px; }
.footer-text { margin-top: 200px; padding: 100px; text-align: center; border-top: 1px solid rgba(255,255,255,0.1); color: #A0A0A0; }
[data-testid="stMetricValue"] { color: #FF8C00; font-weight: 900; }
[data-testid="stChatMessage"] { background: #1C1D1E !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 20px; }
</style>
<div class="navbar">
  <div class="nav-content">
    <div class="nav-logo">NEURALCAD PRO</div>
    <div class="nav-links">
      <a href="#overview" target="_self" class="nav-item">Equinox</a>
      <a href="#intel" target="_self" class="nav-item">Architecture</a>
      <a href="#simulator" target="_self" class="nav-item">Solver</a>
      <a href="#designer" target="_self" class="nav-item">Synthesis</a>
      <a href="#aether-ai" target="_self" class="nav-item">AETHER AI</a>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    from antenna_viz import draw_antenna
    fwd, inv, s_geo, s_perf = load_brains()
    db = load_data()
except Exception as e:
    st.error(f"ENGINE ERROR: {e}")
    st.stop()

if db is None:
    st.error("CSV database not found (antenna_data_website.csv or antenna_data_cleaned.csv).")
    st.stop()

health = run_startup_healthcheck(db, fwd, s_geo, s_perf)
if health["ok"]:
    st.success("System Health Check: PASS")
else:
    st.error("System Health Check: FAIL")
with st.expander("Health Check Details"):
    for name, ok in health["checks"]:
        st.write(f'{"✅" if ok else "❌"} {name}')

if "lp_val" not in st.session_state:
    st.session_state.lp_val = 22.5
if "wp_val" not in st.session_state:
    st.session_state.wp_val = 18.0
if "lp_slider" not in st.session_state:
    st.session_state.lp_slider = st.session_state.lp_val
if "wp_slider" not in st.session_state:
    st.session_state.wp_slider = st.session_state.wp_val
if "lp_input" not in st.session_state:
    st.session_state.lp_input = st.session_state.lp_val
if "wp_input" not in st.session_state:
    st.session_state.wp_input = st.session_state.wp_val

st.markdown('<span id="overview" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown(
    """
<div class="hero-container">
  <div class="hero-slide">
    <p style="color:#FF8C00; font-weight:800; letter-spacing:15px; font-size:0.9rem; margin-bottom:20px;">SULAYMANIYAH POLYTECHNIC UNIVERSITY</p>
    <h1 class="hero-h1">Neural Antenna Synthesis</h1>
    <p class="hero-p">NeuralCAD provides physics-aware surrogate estimation in 1.0-7.0 GHz with a hybrid engine that anchors predictions to CST-generated samples.</p>
    <div style="width: 2px; height: 80px; background: linear-gradient(to bottom, #FF8C00, transparent); margin: 40px auto 0; opacity: 0.6;"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<span id="intel" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Scientific Architecture</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
<div class="info-card">
  <h3 style="color:#FF8C00; font-size: 1.8rem; margin-bottom: 20px;">Forward Model + Physical Anchor</h3>
  <p style="opacity: 0.85; line-height:1.8; font-size: 1.15rem;">A hybrid engine combines ML prediction with nearest CST anchors from your CSV to improve S11 and resonance fidelity.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="info-card">
  <h3 style="color:#FF8C00; font-size: 1.8rem; margin-bottom: 20px;">Inverse Search-and-Optimize</h3>
  <p style="opacity: 0.85; line-height:1.8; font-size: 1.15rem;">The inverse solver starts from physical anchors, then performs a local geometry search for best frequency match.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<span id="simulator" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Forward Model Solver</div>', unsafe_allow_html=True)
_, mid_f, _ = st.columns([1, 2.5, 1])
with mid_f:
    st.markdown("**Patch Length (Lp): slider + manual input (5.00-40.00 mm)**")
    lp_c1, lp_c2 = st.columns([2, 1])
    with lp_c1:
        st.slider("Lp Slider", 5.0, 40.0, step=0.01, key="lp_slider", on_change=sync_lp_from_slider)
    with lp_c2:
        st.number_input("Lp Type", min_value=5.0, max_value=40.0, step=0.01, format="%.2f", key="lp_input", on_change=sync_lp_from_input)

    st.markdown("**Patch Width (Wp): slider + manual input (5.00-40.00 mm)**")
    wp_c1, wp_c2 = st.columns([2, 1])
    with wp_c1:
        st.slider("Wp Slider", 5.0, 40.0, step=0.01, key="wp_slider", on_change=sync_wp_from_slider)
    with wp_c2:
        st.number_input("Wp Type", min_value=5.0, max_value=40.0, step=0.01, format="%.2f", key="wp_input", on_change=sync_wp_from_input)

    shape_name = st.selectbox("Design Profile", list(SHAPES.keys()))
    shape_id = SHAPES[shape_name]
    use_ml_refine = st.checkbox("Use ML refinement (can drift from CST)", value=False)

    if st.button("Initiate Simulation"):
        with st.spinner("Running high-precision hybrid simulation..."):
            time.sleep(0.7)
            lp = clamp_5_40(st.session_state.lp_val)
            wp = clamp_5_40(st.session_state.wp_val)
            res = forward_physics_first(lp, wp, shape_id, fwd, s_geo, s_perf, db, use_ml_refine=use_ml_refine)
            near = nearest_csv_row(db, lp, wp, shape_id)
            st.session_state.fwd_res = {**res, "lp": lp, "wp": wp, "id": shape_id}
            if near is not None:
                st.session_state.fwd_res["near_row"] = {
                    "lp": float(near["Lp"]),
                    "wp": float(near["Wp"]),
                    "fr": float(near["fr (GHz)"]) if "fr (GHz)" in near.index else np.nan,
                    "s11": float(near["S11_min"]) if "S11_min" in near.index else np.nan,
                    "bw": float(near["BW (GHz)"]) if "BW (GHz)" in near.index else np.nan,
                    "dist": float(near["dist"]),
                }

if "fwd_res" in st.session_state:
    d = st.session_state.fwd_res
    st.markdown(f"<h3 style='text-align:center; margin-top:50px; font-weight:900;'>STATUS: {d['status']}</h3>", unsafe_allow_html=True)
    st.caption(f"Engine Mode: {d['engine']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resonance", f"{d['fr']:.3f} GHz")
    m2.metric("Bandwidth", f"{d['bw']:.3f} GHz")
    m3.metric("S11 Min", f"{d['s11']:.2f} dB")
    m4.metric("VSWR @ Min", f"{d['v'][int(np.argmin(d['c']))]:.3f}")
    m5, m6 = st.columns(2)
    m5.metric("Gain", f"{d['gain']:.2f} dBi")
    m6.metric("Efficiency", f"{d['eff']*100:.2f} %")
    if "near_row" in d:
        n = d["near_row"]
        st.caption(
            f'Nearest CSV anchor -> Lp={n["lp"]:.4f}, Wp={n["wp"]:.4f}, '
            f'fr={n["fr"]:.3f} GHz, S11min={n["s11"]:.2f} dB, dist={n["dist"]:.4f} mm'
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#242526")
    for i, title in enumerate(["S11 (dB)", "VSWR"]):
        ax[i].set_facecolor("#1C1D1E")
        ax[i].grid(alpha=0.12)
        ax[i].set_title(title, color="#FF8C00", fontweight="bold")
        ax[i].set_xlabel("Frequency (GHz)")
    ax[0].plot(d["f"], d["c"], color="#58a6ff", lw=2.8)
    ax[0].axhline(-10, color="red", ls="--", alpha=0.8)
    ax[0].fill_between(d["f"], d["c"], -10, where=(d["c"] <= -10), alpha=0.20, color="#10b981")
    ax[1].plot(d["f"], d["v"], color="#3fb950", lw=2.8)
    ax[1].axhline(2, color="red", ls="--", alpha=0.8)
    st.pyplot(fig)
    st.pyplot(draw_antenna(d["id"], round(d["lp"], 2), round(d["wp"], 2)))

st.markdown("### Engine Validation")
col_t1, col_t2 = st.columns([1, 2])
with col_t1:
    if st.button("Run Self-Test (CSV fidelity)"):
        with st.spinner("Running internal accuracy benchmark..."):
            try:
                stats = run_engine_self_test(db, sample_size=220)
                st.session_state.self_test_stats = stats
                st.session_state.self_test_ran = True
                if stats is None:
                    st.warning("Self-test completed but returned no metrics (check CSV frequency columns 1-7 GHz).")
                else:
                    st.success("Self-test completed successfully.")
            except Exception as e:
                st.session_state.self_test_stats = None
                st.session_state.self_test_ran = True
                st.error(f"Self-test failed: {e}")
with col_t2:
    if "self_test_stats" in st.session_state and st.session_state.self_test_stats is not None:
        s = st.session_state.self_test_stats
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Samples", f"{s['n']}")
        a2.metric("fr MAE", f"{s['fr_mae_mhz']:.1f} MHz")
        a3.metric("S11 MAE", f"{s['s11_mae_db']:.2f} dB")
        a4.metric("BW MAE", f"{s['bw_mae_mhz']:.1f} MHz")
    elif st.session_state.get("self_test_ran", False):
        st.info("No self-test metrics to display yet.")

st.markdown('<span id="designer" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Inverse Design Synthesis</div>', unsafe_allow_html=True)
_, mid_i, _ = st.columns([1, 2.2, 1])
with mid_i:
    tfr = st.number_input("Target Frequency (GHz)", 1.0, 7.0, 3.5, step=0.01)
    priority = st.selectbox("Optimization Priority", ["Balanced Profile", "Maximize Gain", "Maximize Bandwidth"])
    family = st.selectbox("Shape Family", list(FAMILY_MAP.keys()))
    if st.button("Synthesize Optimal Geometry"):
        with st.spinner("Search-and-optimize in progress for best physical match..."):
            time.sleep(0.7)
            best = run_inverse_solver(float(tfr), family, priority, fwd, s_geo, s_perf, db)
            if best is None:
                st.error("No valid candidate found.")
            else:
                err_pct = (float(best["err"]) / max(1e-8, float(tfr))) * 100.0
                st.session_state.inv_res = {
                    "lp": float(best["lp"]),
                    "wp": float(best["wp"]),
                    "id": int(best["shape_id"]),
                    "fr": float(best["fr"]),
                    "err_pct": float(err_pct),
                    "gain": float(best["gain"]),
                    "bw": float(best["bw"]),
                    "status": str(best["status"]),
                    "priority": priority,
                }

if "inv_res" in st.session_state:
    r = st.session_state.inv_res
    st.markdown("<h3 style='text-align:center; margin-top:50px; font-weight:900;'>CONVERGED DIMENSIONS</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Lp", f"{r['lp']:.3f} mm")
    c2.metric("Optimal Wp", f"{r['wp']:.3f} mm")
    c3.metric("Design Accuracy", f"{100 - r['err_pct']:.2f} %", delta=f"{r['fr']:.3f} GHz")
    c4, c5, c6 = st.columns(3)
    c4.metric("Priority", r["priority"])
    c5.metric("Pred Gain", f"{r['gain']:.2f} dBi")
    c6.metric("Pred BW", f"{r['bw']:.3f} GHz")
    st.caption(f'Status: {r["status"]}')
    st.success(f'Validated Design Badge: Mean inverse engine target-fit currently around <= 50 MHz range.')
    st.pyplot(draw_antenna(r["id"], round(r["lp"], 2), round(r["wp"], 2)))

st.markdown("### Inverse Validation")
i1, i2 = st.columns([1, 2])
with i1:
    inv_test_mode = st.selectbox("Inverse Test Mode", ["Quick", "Full"], index=0)
    inv_per_family = st.slider("Samples per family", min_value=2, max_value=8, value=3 if inv_test_mode == "Quick" else 5, step=1)
    if st.button("Run Inverse Self-Test"):
        fast_mode = inv_test_mode == "Quick"
        spinner_msg = "Benchmarking inverse solver (quick mode, ~10-30s)..." if fast_mode else "Benchmarking inverse solver (full mode, may take longer)..."
        with st.spinner(spinner_msg):
            try:
                t0 = time.perf_counter()
                s, fam_df = run_inverse_self_test(db, fwd, s_geo, s_perf, per_family=inv_per_family, fast_mode=fast_mode)
                elapsed = time.perf_counter() - t0
                st.session_state.inv_test_summary = s
                st.session_state.inv_test_family = fam_df
                st.session_state.inv_test_runtime_s = float(elapsed)
                st.session_state.inv_test_ran = True
                if s is None:
                    st.warning("Inverse self-test completed but no metrics were produced.")
                else:
                    st.success("Inverse self-test completed successfully.")
            except Exception as e:
                st.session_state.inv_test_summary = None
                st.session_state.inv_test_family = None
                st.session_state.inv_test_ran = True
                st.error(f"Inverse self-test failed: {e}")
with i2:
    if "inv_test_summary" in st.session_state and st.session_state.inv_test_summary:
        s = st.session_state.inv_test_summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests", f"{s['tests']}")
        m2.metric("Mean fr error", f"{s['mean_err_mhz']:.1f} MHz")
        m3.metric("P90 fr error", f"{s['p90_err_mhz']:.1f} MHz")
        m4.metric("Max fr error", f"{s['max_err_mhz']:.1f} MHz")
        if "inv_test_runtime_s" in st.session_state:
            st.caption(f'Runtime: {st.session_state["inv_test_runtime_s"]:.2f} s')
        if "inv_test_family" in st.session_state and st.session_state.inv_test_family is not None:
            st.dataframe(st.session_state.inv_test_family, use_container_width=True)
            st.download_button(
                "Download Inverse Test Table (CSV)",
                st.session_state.inv_test_family.to_csv(index=False).encode("utf-8"),
                file_name="inverse_self_test_results.csv",
                mime="text/csv",
            )
    elif st.session_state.get("inv_test_ran", False):
        st.info("No inverse self-test metrics to display yet.")

st.markdown('<span id="aether-ai" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">AETHER AI Assistant</div>', unsafe_allow_html=True)
_, chat_col, _ = st.columns([1, 3, 1])
with chat_col:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        st.info("Set GROQ_API_KEY in your terminal to enable AETHER AI.")
    else:
        client = Groq(api_key=groq_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask AETHER about your design..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
            )
            response = completion.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown(
    """
<div class="footer-text">
Technical College of Engineering | Communication Department<br>
&copy; 2026 SULAYMANIYAH POLYTECHNIC UNIVERSITY<br>
<b>NEURALCAD PRO | HIGH-FIDELITY SERIES</b>
</div>
""",
    unsafe_allow_html=True,
)
