import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 1. Configuration
API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="M-Pesa Liquidity Tracker", layout="wide")

st.title(" M-Pesa Liquidity Forecasting Dashboard")
st.markdown("---")

# 2. Sidebar for Single Agent Prediction
st.sidebar.header("Individual Agent Lookup")
agent_id = st.sidebar.selectbox("Select Agent", ["Agent_Urban_High", "Agent_Urban_Med", "Agent_Rural_High", "Agent_Rural_Med", "Agent_Rural_Low"])

is_holiday = st.sidebar.checkbox("Is Tomorrow a Holiday?")
is_payday = st.sidebar.checkbox("Is it a Payday Window?")
is_school = st.sidebar.checkbox("Is it a School Fee Window?")
is_weekend = st.sidebar.checkbox("Is it a Weekend?")

if st.sidebar.button("Get Prediction"):
    payload = {
        "agent_id": agent_id,
        "is_holiday": int(is_holiday),
        "is_payday_window": int(is_payday),
        "is_school_window": int(is_school),
        "is_weekend": int(is_weekend)
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        st.sidebar.success(f"Advice: {data['action']}")
        st.sidebar.metric("Predicted Need", f"{data['prediction_kes']:,.2f} KES")
        st.sidebar.metric("Safety Max", f"{data['safety_buffer_high']:,.2f} KES")
    else:
        st.sidebar.error("Could not reach API. Is it running?")

# 3. Main Dashboard: Network Summary
st.subheader(" Network-Wide Forecast (Normal Day)")

if st.button("Refresh Network Status"):
    response = requests.get(f"{API_URL}/network_summary")
    
    if response.status_code == 200:
        report_data = response.json()["agents"]
        df = pd.DataFrame(report_data)
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Agents", len(df))
        col2.metric("Total Expected Outflow", f"{df['expected_need'].sum():,.2f} KES")
        high_risk_count = len(df[df['risk'] == 'High'])
        col3.metric("High Risk Agents", high_risk_count, delta=high_risk_count, delta_color="inverse")

        # Visualizations
        st.markdown("### Liquidity Needs by Agent")
        fig = px.bar(df, x='agent', y='expected_need', color='risk',
                     title="Expected Net Liquidity Needs",
                     labels={'expected_need': 'KES Needed', 'agent': 'Agent Location'},
                     color_discrete_map={'High': 'red', 'Normal': 'green'})
        st.plotly_chart(fig, use_container_width=True)

        st.table(df)
    else:
        st.error("API is offline. Run 'uvicorn Scripts.main:app' first!")

st.markdown("---")
st.caption("Developed by Reinhardt Kiage | Petroleum & ML Engineer")
