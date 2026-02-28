import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
try:
    from fpdf import FPDF
except ImportError:
    st.error("Please run 'pip install fpdf' in your terminal!")

# 1. Load the model and scaler
try:
    model = joblib.load('battery_usability_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Files loading error: {e}")

# Page Setup
st.set_page_config(page_title="Battery Health Prediction", layout="wide")
st.title("🔋 Battery Health Prediction System") # Topic එක වෙනස් කළා
st.markdown("---")

if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for Inputs
st.sidebar.header("📥 Sensor Data Input")
v = st.sidebar.number_input("Voltage (V)", value=52.0, step=0.1)
i = st.sidebar.number_input("Current (A)", value=2.0, step=0.1)
t = st.sidebar.number_input("Temperature (°C)", value=30.0, step=0.1)
power = round(v * i, 2)

# Logic to define suitability, remaining life and load
def get_detailed_prediction(score):
    if score >= 80:
        return "2 - 3 Years", "High (Up to 500W)", "Excellent (Solar/UPS/EV)"
    elif score >= 50:
        return "1 - 2 Years", "Medium (Up to 150W)", "Good (LED/Fans/Small Electronics)"
    elif score >= 30:
        return "6 Months - 1 Year", "Low (Below 50W)", "Fair (Emergency backup only)"
    else:
        return "0 Months", "No Load (Recycle)", "Critical (Not safe for use)"

if st.sidebar.button("⚡ Predict Health & Usage"):
    input_array = np.array([[v, i, t, power]])
    scaled_data = scaler.transform(input_array)
    prediction = model.predict(scaled_data)[0]
    score = round(float(prediction), 2)
    
    rem_life, rec_load, suitability = get_detailed_prediction(score)
    
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Voltage": v, "Current": i, "Temp": t, "Power": power, 
        "Score": score, "Rem_Life": rem_life, "Rec_Load": rec_load, "Suitability": suitability
    })

# Main Dashboard
if st.session_state.history:
    last = st.session_state.history[-1]
    
    # Prediction results display
    st.subheader(f"Status: {last['Suitability']}")
    
    res1, res2, res3 = st.columns(3)
    res1.metric("Estimated Remaining Life", last['Rem_Life'])
    res2.metric("Recommended Load", last['Rec_Load'])
    res3.metric("Health Score", f"{last['Score']}%")

    st.markdown("---")

    # Graphs Row
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("🎯 Health Gauge")
        fig_health = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = last['Score'],
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 30], 'color': "#ff4b4b"},
                    {'range': [30, 60], 'color': "#ffa500"},
                    {'range': [60, 100], 'color': "#28a745"}]
            }
        ))
        st.plotly_chart(fig_health, use_container_width=True)

    with g2:
        st.subheader("🧊 3D Parameter Analysis") # ඔයා ඉල්ලපු 3D Plot එක
        df = pd.DataFrame(st.session_state.history)
        fig_3d = px.scatter_3d(df, x='Voltage', y='Current', z='Score', color='Temp', size='Power')
        st.plotly_chart(fig_3d, use_container_width=True)

    # Health vs Load Relationship Graph
    st.subheader("📈 Suitability Overview")
    fig_suit = px.bar(df, x="Time", y="Score", color="Score", 
                     hover_data=['Rem_Life', 'Rec_Load'],
                     color_continuous_scale='RdYlGn', title="Prediction Consistency over Session")
    st.plotly_chart(fig_suit, use_container_width=True)

    # PDF Generation
    def export_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(190, 15, "Battery Health Prediction Report", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(190, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(20, 10, "Time", 1)
        pdf.cell(20, 10, "Health %", 1)
        pdf.cell(40, 10, "Est. Life", 1)
        pdf.cell(40, 10, "Rec. Load", 1)
        pdf.cell(70, 10, "Suitability", 1)
        pdf.ln()

        pdf.set_font("Arial", size=8)
        for _, row in dataframe.iterrows():
            pdf.cell(20, 10, str(row['Time']), 1)
            pdf.cell(20, 10, f"{row['Score']}%", 1)
            pdf.cell(40, 10, str(row['Rem_Life']), 1)
            pdf.cell(40, 10, str(row['Rec_Load']), 1)
            pdf.cell(70, 10, str(row['Suitability']), 1)
            pdf.ln()
        
        return pdf.output(dest='S').encode('latin-1')

    # PDF Download Button
    st.markdown("---")
    try:
        pdf_bytes = export_pdf(df)
        st.download_button(label="📄 Download Detailed PDF Report", data=pdf_bytes, 
                           file_name="Battery_Prediction_Report.pdf", mime="application/pdf")
    except:
        st.warning("Install 'fpdf' to enable PDF downloads.")

# Data Logs
if st.session_state.history:
    st.subheader("📋 Session History Logs")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)