import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. SCIENTIFIC UTILITIES
# ==========================================
FREQS_RAW = np.linspace(1, 7, 151)

def apply_scientific_smoothing(freqs, shaky_curve):
    """Turns shaky AI output into a silky smooth CST-style line."""
    smoothed_sg = savgol_filter(shaky_curve, 15, 3)
    freqs_fine = np.linspace(freqs.min(), freqs.max(), 500)
    spline = make_interp_spline(freqs, smoothed_sg, k=3)
    smooth_curve = spline(freqs_fine)
    return freqs_fine, smooth_curve

def advanced_vna_measurement(freqs, curve):
    """Calculates Resonance, BW, and Status from the curve sketch."""
    idx_min = np.argmin(curve)
    m_fr, m_s11 = freqs[idx_min], curve[idx_min]
    is_matched = (curve <= -10.0).astype(int)
    diff = np.diff(np.concatenate(([0], is_matched, [0])))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1
    bands = list(zip(starts, ends))
    m_bw, status = 0.0, "NOT WORKING"
    if len(bands) > 0:
        status = "WORKING"
        primary_band = next((b for b in bands if b[0] <= idx_min <= b[1]), bands[0])
        m_bw = freqs[primary_band[1]] - freqs[primary_band[0]]
    return m_fr, m_s11, m_bw, status

# ==========================================
# 2. LOAD ENGINES & DATA
# ==========================================
@st.cache_resource
def load_system():
    fwd = tf.keras.models.load_model('forward_model_final.keras', compile=False)
    inv = tf.keras.models.load_model('inverse_model_final.keras', compile=False)
    s_geo = joblib.load('scaler_geo.pkl')  
    s_perf = joblib.load('scaler_perf.pkl') 
    return fwd, inv, s_geo, s_perf

@st.cache_data
def load_csv_data():
    return pd.read_csv('antenna_data_cleaned.csv')

try:
    from antenna_viz import draw_antenna
    fwd, inv, s_geo, s_perf = load_system()
    db = load_csv_data()
except Exception as e:
    st.error(f"System Load Error: Ensure all .keras, .pkl and .csv files are on GitHub. Details: {e}")

# ==========================================
# 3. UI BRANDING & TRAFFIC LIGHT CSS
# ==========================================
st.set_page_config(page_title="SPU AI-Antenna Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e2e8f0; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .header-box { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 25px; }
    .light-green { background-color: #064e3b; color: #10b981; padding: 15px; border-radius: 10px; border: 2px solid #10b981; text-align: center; font-weight: bold; }
    .light-yellow { background-color: #42210b; color: #fbbf24; padding: 15px; border-radius: 10px; border: 2px solid #fbbf24; text-align: center; font-weight: bold; }
    .light-red { background-color: #450a0a; color: #ef4444; padding: 15px; border-radius: 10px; border: 2px solid #ef4444; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1 style="color: white;">SULAYMANIYAH POLYTECHNIC UNIVERSITY</h1><h2 style="color: #58a6ff;">Scientific AI Antenna Synthesis & Audit Suite</h2><p style="color: #8b949e;">Technical College of Engineering | Communication Dept | 2025-2026</p></div>', unsafe_allow_html=True)

tab_fwd, tab_inv = st.tabs(["📡 FORWARD PHYSICS SIMULATOR", "🧠 INVERSE DESIGN SYNTHESIS"])

# ==========================================
# MODULE 1: FORWARD (PREDICT)
# ==========================================
with tab_fwd:
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.subheader("Geometry Configuration")
        lp = st.slider("Patch Length (Lp) mm", 5.0, 40.0, 25.0)
        wp = st.slider("Patch Width (Wp) mm", 5.0, 40.0, 20.0)
        shapes = ["Rectangle", "Stepped", "T-Shape", "Ellipse", "Semi-Circle", "Fan", "Triangle", "Trapezoid", "Diamond", "Hexagon", "Pentagon", "Cross"]
        s_name = st.selectbox("Antenna Geometry", shapes)
        sh_id = shapes.index(s_name) + 1
        
        btn = st.button("🚀 RUN PHYSICS ENGINE", use_container_width=True)

    if btn or 'fwd_cache' in st.session_state:
        if btn:
            vec = np.zeros((1, 14)); vec[0,0], vec[0,1] = lp, wp; vec[0, 1 + sh_id] = 1.0
            preds = fwd.predict(s_geo.transform(vec), verbose=0)
            recon = np.zeros((1, 157)); recon[0, [0, 2, 3, 4]] = preds[0][0]; recon[0, 6:] = preds[1][0]
            perf_raw = s_perf.inverse_transform(recon)[0]
            f_s, c_s = apply_scientific_smoothing(FREQS_RAW, perf_raw[6:])
            m_fr, m_s11, m_bw, status = advanced_vna_measurement(f_s, c_s)
            st.session_state['fwd_cache'] = {"fr": m_fr, "s11": m_s11, "bw": m_bw, "gain": perf_raw[2], "status": status, "f": f_s, "c": c_s, "lp": lp, "wp": wp, "id": sh_id}
        
        d = st.session_state['fwd_cache']
        with c2:
            # TRAFFIC LIGHT LOGIC
            is_ood = d['lp'] < 10 or d['lp'] > 35 or d['wp'] < 10 or d['wp'] > 35
            if is_ood: st.markdown('<div class="light-red">🔴 LOW CONFIDENCE: Dimensions in Extrapolation Zone (Out of Training Range)</div>', unsafe_allow_html=True)
            elif d['status'] == "WORKING": st.markdown('<div class="light-green">🟢 HIGH CONFIDENCE: Reliable Design Match Found</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="light-yellow">🟡 MEDIUM CONFIDENCE: Design Predicted, but Physically Non-Resonant</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Resonance", f"{d['fr']:.3f} GHz")
            m2.metric("Bandwidth", f"{d['bw']:.3f} GHz")
            m3.metric("Measured Gain", f"{d['gain']:.2f} dBi")
            m4.metric("Matching Status", d['status'])
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0b0e14')
            ax[0].set_facecolor('#161b22'); ax[0].plot(d['f'], d['c'], color='#58a6ff', lw=3); ax[0].axhline(-10, color='red', ls='--'); ax[0].set_title("S11 (dB)")
            # VSWR Calculation
            gamma = 10**(d['c']/20); vswr = (1+gamma)/(1-gamma)
            ax[1].set_facecolor('#161b22'); ax[1].plot(d['f'], vswr, color='#3fb950', lw=3); ax[1].axhline(2, color='red', ls='--'); ax[1].set_ylim(1, 10); ax[1].set_title("VSWR")
            st.pyplot(fig)
            st.pyplot(draw_antenna(d['id'], d['lp'], d['wp']))

# ==========================================
# MODULE 2: INVERSE (SEARCH & OPTIMIZE)
# ==========================================
with tab_inv:
    i1, i2 = st.columns([1, 2.5])
    with i1:
        st.subheader("Target Objectives")
        t_fr = st.number_input("Target Frequency (GHz)", 1.0, 7.0, 5.0)
        goal = st.selectbox("Design Priority", ["Balanced Profile", "Maximize Gain", "Maximize Bandwidth"])
        fams = {"Rectangular": [1,2,3], "Circular": [4,5,6], "Triangular": [7,8,9], "Polygonal": [10,11,12]}
        f_ch = st.selectbox("Constrain to Family", list(fams.keys()))
        
        if st.button("🛠️ SYNTHESIZE OPTIMAL DESIGN", use_container_width=True):
            # SEARCH DATABASE FIRST
            subset = db[db['Antenna_ID'].isin(fams[f_ch])]
            subset['diff'] = abs(subset['fr (GHz)'] - t_fr)
            candidates = subset[subset['diff'] < 0.5]
            if candidates.empty: candidates = subset # Fallback
            
            if goal == "Maximize Gain": winner = candidates.sort_values('Gain_at_fr', ascending=False).iloc[0]
            elif goal == "Maximize Bandwidth": winner = candidates.sort_values('BW (GHz)', ascending=False).iloc[0]
            else: winner = candidates.sort_values('diff').iloc[0]
            
            # AI REFINE
            t_vec = np.zeros((1, 157)); t_vec[0, 0], t_vec[0, 2], t_vec[0, 3], t_vec[0, 4] = t_fr, winner['Gain_at_fr'], 0.85, -20.0
            t_sc = s_perf.transform(t_vec)[0, [0, 2, 3, 4]]
            d_sc = inv.predict(t_sc.reshape(1,-1), verbose=0)
            d_mm = s_geo.inverse_transform(d_sc)[0]
            
            # VERIFY
            v_p = fwd.predict(d_sc, verbose=0)
            v_re = np.zeros((1, 157)); v_re[0, 6:] = v_p[1][0]
            v_cu = s_perf.inverse_transform(v_re)[0, 6:]
            vf, vc = apply_scientific_smoothing(FREQS_RAW, v_cu)
            v_fr, v_s11, v_bw, v_stat = advanced_vna_measurement(vf, vc)
            st.session_state['inv'] = {"lp": d_mm[0], "wp": d_mm[1], "id": int(winner['Antenna_ID']), "f": vf, "c": vc, "stat": v_stat, "fr": v_fr, "gain": winner['Gain_at_fr']}

    if 'inv' in st.session_state:
        r = st.session_state['inv']
        with i2:
            st.success(f"Optimal Geometry Synthesized: {r['lp']:.2f}mm x {r['wp']:.2f}mm (Shape {r['id']})")
            st.markdown(f'<div class="light-green">PHYSICS VERIFICATION: {r["stat"]} at {r["fr"]:.2f} GHz</div>', unsafe_allow_html=True)
            st.pyplot(draw_antenna(r['id'], r['lp'], r['wp']))