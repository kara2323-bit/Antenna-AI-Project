import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Antenna Designer", layout="wide")

# --- MODERN CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
    .metric-card { 
        background-color: #1e293b; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center;
        border: 1px solid #334155;
    }
    .hero-section { 
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%); 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_brains():
    try:
        fwd_m = tf.keras.models.load_model('forward_model_final.keras', compile=False)
        inv_m = tf.keras.models.load_model('inverse_model_final.keras', compile=False)
        s_g = joblib.load('scaler_geo.pkl')
        s_p = joblib.load('scaler_perf.pkl')
        return fwd_m, inv_m, s_g, s_p
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Assign to global variables
fwd, inv, s_geo, s_perf = load_brains()

# --- HEADER ---
st.markdown("""
    <div class="hero-section">
        <h1 style='color: white; margin:0;'>AI ANTENNA DESIGNER</h1>
        <p style='color: #e2e8f0;'>Machine Learning Approach for CPW Microstrip Patch Antennas</p>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
app_mode = st.sidebar.selectbox("Select Mode", ["Forward Predictor", "Inverse Designer"])

if fwd is None:
    st.warning("Models are still loading or files are missing from GitHub. Please check the 'Manage App' logs.")
    st.stop()

# --- MODE 1: FORWARD ---
if app_mode == "Forward Predictor":
    st.subheader("📡 Performance Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Geometry Inputs")
        lp_val = st.slider("Patch Length (Lp) mm", 10.0, 35.0, 22.5)
        wp_val = st.slider("Patch Width (Wp) mm", 10.0, 35.0, 20.0)
        
        shapes = {
            "Rectangle": 1, "Stepped": 2, "T-Shape": 3, "Ellipse": 4, 
            "Semi-circle": 5, "Fan": 6, "Triangle": 7, "Trapezoid": 8, 
            "Rhombus": 9, "Hexagon": 10, "Pentagon": 11, "Cross": 12
        }
        shape_choice = st.selectbox("Select Antenna Shape", list(shapes.keys()))
        s_id = shapes[shape_choice]
        
        # --- THE BUTTON YOU NEED ---
        predict_btn = st.button("🚀 Predict Performance", use_container_width=True)

    with col2:
        if predict_btn:
            # Preparing input vector
            geo_in = np.zeros((1, 14))
            geo_in[0, 0] = lp_val
            geo_in[0, 1] = wp_val
            geo_in[0, 1 + s_id] = 1.0
            
            # AI Inference
            geo_scaled = s_geo.transform(geo_in)
            preds = fwd.predict(geo_scaled, verbose=0)
            
            # Unscaling
            recon = np.hstack([preds[0], preds[1]])
            perf = s_perf.inverse_transform(recon)[0]
            
            # Results UI
            r1, r2, r3 = st.columns(3)
            r1.markdown(f'<div class="metric-card">Freq<br><h2>{perf[0]:.3f} GHz</h2></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="metric-card">Gain<br><h2>{perf[2]:.2f} dBi</h2></div>', unsafe_allow_html=True)
            r3.markdown(f'<div class="metric-card">Eff<br><h2>{perf[3]*100:.1f} %</h2></div>', unsafe_allow_html=True)
            
            # S11 Graph
            st.markdown("### S11 Reflection Coefficient")
            fig, ax = plt.subplots(figsize=(10, 4))
            f_ax = np.linspace(1, 7, 151)
            # Apply smoothing to make it look professional
            curve = savgol_filter(perf[6:], 11, 3)
            ax.plot(f_ax, curve, color='#6366f1', lw=3)
            ax.axhline(-10, color='red', linestyle='--', alpha=0.5)
            ax.set_facecolor('#0e1117')
            ax.grid(color='#334155', linestyle='--', alpha=0.3)
            ax.set_ylim(-40, 5)
            st.pyplot(fig)
        else:
            st.info("Adjust the sliders on the left and click 'Predict Performance' to see the results.")

# --- MODE 2: INVERSE ---
else:
    st.subheader("🤖 AI Inverse Designer")
    tr_fr = st.number_input("Target Frequency (GHz)", 1.0, 7.0, 5.0)
    tr_gn = st.number_input("Target Gain (dBi)", -5.0, 10.0, 3.0)
    
    if st.button("🛠️ Generate Optimized Design", use_container_width=True):
        # Target prep
        t_full = np.zeros((1, 157))
        t_full[0, 0] = tr_fr
        t_full[0, 2] = tr_gn
        t_full[0, 4] = -20.0 # Standard match target
        
        t_scaled = s_perf.transform(t_full)[0, :6]
        d_scaled = inv.predict(t_scaled.reshape(1, -1), verbose=0)
        d = s_geo.inverse_transform(d_scaled)[0]
        
        best_shape_idx = np.argmax(d[2:]) + 1
        
        c1, c2 = st.columns(2)
        c1.success(f"Recommended Lp: {d[0]:.2f} mm")
        c2.success(f"Recommended Wp: {d[1]:.2f} mm")
        st.info(f"Recommended Shape: ID {best_shape_idx}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>SPU Technical College of Engineering | Communication Dept</p>", unsafe_allow_html=True)
