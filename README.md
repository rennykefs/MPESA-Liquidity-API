#  M-Pesa Liquidity Forecasting System
**An End-to-End AI Solution for Mobile Money Agent Optimization**

###  Project Overview
This system predicts daily cash requirements for M-Pesa agents in Kenya. Using **SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with Exogenous Factors)**, it accounts for payday cycles, holidays, and school fee windows to prevent "stock-outs" and optimize cash-in-transit routes.

###  Architecture
- **Backend:** FastAPI (Python) - High-performance API serving models from RAM.
- **Frontend:** Streamlit - Interactive dashboard for bank operations managers.
- **Model:** SARIMAX time-series forecasting with `joblib` serialization.

###  Quick Start
1. **Clone the repo:** `git clone https://github.com/YOUR_USERNAME/mpesa-liquidity.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the API:** `uvicorn Scripts.main:app --reload`
4. **Run the Dashboard:** `streamlit run dashboard.py`

###  Features
- **Real-time Prediction:** Get instant "Withdraw/Deposit" advice for specific agents.
- **Network Summary:** High-level view of liquidity risk across all agent categories (Urban/Rural).
- **Dynamic Scaling:** Automated model loading using Python Lifespan events.