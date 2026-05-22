import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# --- IMPORT YOUR DRAWING ENGINE ---
try:
    from antenna_viz import draw_antenna
except ImportError:
    st.error("antenna_viz.py not found in folder!")

# ==========================================
# PAGE CONFIG & MODERN THEME
# ==========================================
st.set_page_config(page_title="AI Antenna Designer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #4f46e5; color: white; font-weight: bold; }
    .metric-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #4f46e5; margin-bottom: 10px; }
    .hero-section { background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 100%); color: white; padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# LOAD AI MODELS
# ==========================================
@st.cache_resource
def load_brains():
    # Adding compile=False and safe_mode=False ensures compatibility across versions
    fwd = tf.keras.models.load_model('forward_model_final.keras', compile=False)
    inv = tf.keras.models.load_model('inverse_model_final.keras', compile=False)
    s_geo = joblib.load('scaler_geo.pkl')
    s_perf = joblib.load('scaler_perf.pkl')
    return fwd, inv, s_geo, s_perf

# ==========================================
# UI HEADER
# ==========================================
st.markdown("""
    <div class="hero-section">
        <h1>SULAYMANIYAH POLYTECHNIC UNIVERSITY</h1>
        <h3>Machine Learning Approach in Design Microstrip Antenna</h3>
        <p>Technical College of Engineering - Communication Department</p>
    </div>
    """, unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Select Mode", ["Forward Predictor", "Inverse Designer"])

# ==========================================
# MODE 1: FORWARD PREDICTOR
# ==========================================
if app_mode == "Forward Predictor":
    st.title("📡 Forward Performance Predictor")
    
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    
    with col1:
        st.subheader("Control Panel")
        lp = st.slider("Patch Length (Lp) mm", 10.0, 35.0, 22.5)
        wp = st.slider("Patch Width (Wp) mm", 10.0, 35.0, 20.0)
        shape_options = {
            "Rectangle": 1, "Stepped": 2, "T-Shape": 3, "Ellipse": 4, 
            "Semi-circle": 5, "Pie": 6, "Triangle": 7, "Trapezoid": 8, 
            "Rhombus": 9, "Hexagon": 10, "Pentagon": 11, "Cross": 12
        }
        shape_name = st.selectbox("Select Antenna Shape", list(shape_options.keys()))
        id_num = shape_options[shape_name]

        # Prepare Input
        geo_in = np.zeros((1, 14))
        geo_in[0, 0] = lp
        geo_in[0, 1] = wp
        geo_in[0, 1 + id_num] = 1.0
        
        # Predict
        geo_scaled = s_geo.transform(geo_in)
        preds = fwd.predict(geo_scaled, verbose=0)
        perf = s_perf.inverse_transform(np.hstack([preds[0], preds[1]]))[0]

    with col2:
        st.subheader("Antenna Geometry")
        # --- DYNAMICALLY DRAW THE SHAPE ---
        fig_viz = draw_antenna(id_num, lp, wp)
        st.pyplot(fig_viz)
        
        # Performance Metrics
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-card"><b>Resonance</b><br><h3>{perf[0]:.3f} GHz</h3></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><b>S11 Depth</b><br><h3>{perf[4]:.1f} dB</h3></div>', unsafe_allow_html=True)

    with col3:
        st.subheader("S11 Reflection Coefficient")
        fig, ax = plt.subplots()
        f_ax = np.linspace(1, 7, 151)
        curve = savgol_filter(perf[6:], 11, 3)
        ax.plot(f_ax, curve, color='#4f46e5', lw=2.5)
        ax.axhline(-10, color='red', linestyle='--')
        ax.set_ylabel("S11 (dB)")
        ax.set_xlabel("Frequency (GHz)")
        st.pyplot(fig)

# ==========================================
# MODE 2: INVERSE DESIGNER
# ==========================================
else:
    st.title("🤖 AI Inverse Designer")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Desired Performance")
        tr_fr = st.number_input("Target Frequency (GHz)", 1.0, 7.0, 5.0)
        tr_gain = st.number_input("Desired Gain (dBi)", -10.0, 10.0, 3.5)
        
        if st.button("Generate Design"):
            t_full = np.zeros((1, 157))
            t_full[0, 0] = tr_fr
            t_full[0, 2] = tr_gain
            t_full[0, 3] = 0.85 # Efficiency
            t_full[0, 4] = -25.0 # S11
            t_full[0, 5] = 1 # Match flag
            
            t_scaled = s_perf.transform(t_full)[0, :6]
            design_scaled = inv.predict(t_scaled.reshape(1, -1), verbose=0)
            design = s_geo.inverse_transform(design_scaled)[0]
            
            st.session_state['inv_design'] = design

    if 'inv_design' in st.session_state:
        d = st.session_state['inv_design']
        best_id = np.argmax(d[2:]) + 1
        with col2:
            st.success(f"AI suggests Shape ID {best_id}")
            fig_inv = draw_antenna(best_id, d[0], d[1])
            st.pyplot(fig_inv)
            st.write(f"**Recommended Dimensions:** Lp = {d[0]:.2f}mm, Wp = {d[1]:.2f}mm")

st.markdown("---")
st.markdown("<center>Prepared by Communication Engineering Team | Academic Year 2025-2026</center>", unsafe_allow_html=True)
