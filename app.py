
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Electronic Component Failure Prediction", page_icon="⚡", layout="wide")

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load("best_failure_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.title("⚡ Electronic Component Failure Prediction")
st.markdown("### Predict failure probability of electronic components")

st.sidebar.header("Input Parameters")

# Categorical Inputs
component_type = st.sidebar.selectbox("Component Type", 
    ['Capacitor', 'IC', 'Transistor', 'Diode', 'Inductor', 'Resistor'])

package_type = st.sidebar.selectbox("Package Type", ['SMD', 'Through-hole', 'BGA'])

manufacturer_grade = st.sidebar.selectbox("Manufacturer Grade", 
    ['Automotive', 'Industrial', 'Commercial', 'Military'])

environment = st.sidebar.selectbox("Environment", 
    ['Indoor', 'Outdoor', 'Industrial', 'Lab', 'Automotive'])

cooling_method = st.sidebar.selectbox("Cooling Method", ['Heatsink', 'Fan', 'None'])

load_type = st.sidebar.selectbox("Load Type", ['Pulsed', 'Continuous', 'Intermittent'])

visual_inspection = st.sidebar.selectbox("Visual Inspection", 
    ['Normal', 'Discoloration', 'Corroded', 'Cracked', 'Bulging'])

# Numerical Inputs
st.sidebar.markdown("---")
operating_temp_C = st.sidebar.slider("Operating Temperature (°C)", 0.0, 130.0, 40.0)
rated_voltage_V = st.sidebar.number_input("Rated Voltage (V)", 1.0, 60.0, 12.0)
applied_voltage_V = st.sidebar.number_input("Applied Voltage (V)", 1.0, 80.0, 12.0)
current_mA = st.sidebar.number_input("Current (mA)", 0.0, 300.0, 100.0)
power_dissipation_mW = st.sidebar.number_input("Power Dissipation (mW)", 0.0, 17000.0, 1500.0)
vibration_g = st.sidebar.slider("Vibration (g)", 0.0, 4.0, 1.0)
humidity_pct = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 55.0)
operating_hours = st.sidebar.number_input("Operating Hours", 0.0, 40000.0, 10000.0)
thermal_cycles = st.sidebar.number_input("Thermal Cycles", 0, 5000, 500)
ripple_pct = st.sidebar.number_input("Ripple (%)", 0.0, 12.0, 3.0)
esr_mohm = st.sidebar.number_input("ESR (mOhm)", 0.0, 160.0, 50.0)
leakage_current_uA = st.sidebar.number_input("Leakage Current (µA)", 0.0, 40.0, 8.0)
resistance_drift_pct = st.sidebar.number_input("Resistance Drift (%)", 0.0, 5.0, 1.5)

# Predict Button
if st.sidebar.button("🔍 Predict Failure", type="primary"):
    
    input_df = pd.DataFrame([{
        'component_type': component_type,
        'package_type': package_type,
        'manufacturer_grade': manufacturer_grade,
        'environment': environment,
        'cooling_method': cooling_method,
        'load_type': load_type,
        'visual_inspection': visual_inspection,
        'operating_temp_C': operating_temp_C,
        'rated_voltage_V': rated_voltage_V,
        'applied_voltage_V': applied_voltage_V,
        'current_mA': current_mA,
        'power_dissipation_mW': power_dissipation_mW,
        'vibration_g': vibration_g,
        'humidity_pct': humidity_pct,
        'operating_hours': operating_hours,
        'thermal_cycles': thermal_cycles,
        'ripple_pct': ripple_pct,
        'esr_mohm': esr_mohm,
        'leakage_current_uA': leakage_current_uA,
        'resistance_drift_pct': resistance_drift_pct
    }])
    
    # Transform and Predict
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]
    
    st.markdown("---")
    
    if prediction == 1:
        st.error(f"### ❌ Prediction: **FAILURE**")
    else:
        st.success(f"### ✅ Prediction: **NO FAILURE**")
    
    st.metric("Failure Probability", f"{probability*100:.2f}%")
    st.progress(float(probability))
    
    st.markdown("### Risk Level")
    if probability >= 0.70:
        st.warning("🔴 **High Risk** — Immediate attention recommended")
    elif probability >= 0.40:
        st.info("🟡 **Moderate Risk** — Monitor the component")
    else:
        st.success("🟢 **Low Risk** — Component is healthy")
