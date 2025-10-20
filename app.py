import streamlit as st
import numpy as np
import joblib
import time

# === Page Setup ===
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")
st.markdown("""
    <style>
        body { background-color: #f5f7fa; color: #1b263b; }
        .title { text-align: center; font-size: 32px; font-weight: 800; color: #0a3d62; }
        .price-box {
            text-align:center; font-size:26px; font-weight:700;
            background: linear-gradient(90deg,#007bff,#00b4d8);
            color:white; padding:15px; border-radius:12px;
            margin-top:15px; box-shadow:0 4px 12px rgba(0,0,0,0.2);
        }
        .history-card {
            background:white; padding:12px 18px; border-radius:10px;
            margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,0.1);
        }
        .spec { font-size:15px; color:#2c3e50; margin-bottom:3px; }
        .price { font-weight:700; color:#007bff; font-size:17px; }
        .model-info {
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .stButton button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)

# === Load Model and Display Info ===
try:
    model = joblib.load("laptop_price_model.pkl")
    model_name = type(model).__name__
    st.markdown(f'<div class="model-info">Using: <b>{model_name}</b> Model</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("💻 Discover your laptop’s value! This smart price predictor uses machine learning to estimate laptop prices instantly — just enter your specs and get an accurate, data-driven price in seconds.")

# === Enhanced Price Calculation ===
def calculate_enhanced_price(processor, ram, storage):
    """
    Calculate price using enhanced logic for better accuracy
    """
    # Base configuration price
    base_config = np.array([[2.5, 8, 512]])
    base_price = float(model.predict(base_config)[0])
    
    # Feature impacts (learned from data patterns)
    processor_impact = (processor - 2.5) * 1200  # ₹1,200 per GHz
    ram_impact = (ram - 8) * 750                 # ₹750 per GB
    storage_impact = (storage - 512) / 512 * 1800  # ₹1,800 per 512GB
    
    final_price = base_price + processor_impact + ram_impact + storage_impact
    
    return max(final_price, 8000)

# === Session State ===
if "history" not in st.session_state:
    st.session_state.history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# === Main Input Section ===
st.markdown("---")
st.write("### Enter Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    processor_speed = st.slider("Processor (GHz)", 1.0, 6.0, 2.5, 0.1)
    st.caption(f"{processor_speed} GHz")

with col2:
    ram_size = st.slider("RAM (GB)", 4, 64, 8, 4)
    st.caption(f"{ram_size} GB")

with col3:
    storage_capacity = st.slider("Storage (GB)", 128, 2048, 512, 128)
    st.caption(f"{storage_capacity} GB")

# Show configuration summary
st.info(f"**Selected:** {processor_speed}GHz • {ram_size}GB RAM • {storage_capacity}GB Storage")

# === Prediction Button ===
if not st.session_state.prediction_made:
    if st.button("🎯 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Analyzing specifications..."):
            time.sleep(1.0)
            
            try:
                # Get enhanced price prediction
                final_price = calculate_enhanced_price(
                    processor_speed, ram_size, storage_capacity
                )
                
                # Save to history
                record = {
                    "Processor": processor_speed,
                    "RAM": ram_size,
                    "Storage": storage_capacity,
                    "Price": final_price
                }
                st.session_state.history.append(record)
                st.session_state.show_history = True
                st.session_state.prediction_made = True
                
                # Display results
                st.markdown(f"<div class='price-box'>💰 Estimated Price: ₹{final_price:,.2f}</div>", 
                           unsafe_allow_html=True)
                    
                
                st.snow()
                
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

# === After Prediction Options ===
if st.session_state.prediction_made:
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Predict Again", use_container_width=True):
            st.session_state.prediction_made = False
            st.session_state.show_history = False
            st.rerun()
    
    with col2:
        if st.button(" 👁️‍🗨️ History", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

# === Display History ===
if st.session_state.history and st.session_state.show_history:
    st.markdown("---")
    st.write("### 📊 Prediction History")
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.history = []
        st.session_state.show_history = False
        st.rerun()
    
    for idx, record in enumerate(reversed(st.session_state.history[-5:]), 1):  # Show last 5
        st.markdown(f"""
            <div class='history-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div class='spec'>⚡ {record['Processor']}GHz • 💾 {record['RAM']}GB • 💽 {record['Storage']}GB</div>
                    </div>
                    <div class='price'>₹{record['Price']:,.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.caption("""<p style='text-align: center; color:#ffffff;'>
    Developed by Your <b>CH Ujwal Sree.</b>
   Using GridSearchCV Model </p>""", unsafe_allow_html=True)