from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from contextlib import asynccontextmanager

# 1. Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_FOLDER = os.path.join(BASE_DIR, "Model")
print(f"---  Project Root: {BASE_DIR}")
print(f"---  Looking for models in: {MODEL_FOLDER}")

MODELS = {}

# 2. Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models 
    print("\n---  Starting Liquidity API ---")
    if os.path.exists(MODEL_FOLDER):
        for file in os.listdir(MODEL_FOLDER):
            if file.endswith(".pkl"):
                # Clean name
                agent_name = file.replace("sarimax_", "").replace(".pkl", "")
                try:
                    full_path = os.path.join(MODEL_FOLDER, file)
                    MODELS[agent_name] = joblib.load(full_path)
                    print(f" Loaded Brain: {agent_name}")
                except Exception as e:
                    print(f" Could not load {agent_name}: {e}")
    else:
        print(f" ERROR: Folder not found at {MODEL_FOLDER}")
    
    print(f"---  Startup Complete: {len(MODELS)} agents online ---\n")
    yield 
    # Shutdown: Clean up
    MODELS.clear()
    print("---  Server Stopped: Memory Cleared ---")

# 3. Initialize App
app = FastAPI(title="M-Pesa Liquidity Prediction API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    agent_id: str
    is_holiday: int
    is_payday_window: int
    is_school_window: int
    is_weekend: int

@app.get("/")
def home():
    return {
        "status": "Online",
        "active_agents": list(MODELS.keys()),
        "message": "Send a POST request to /predict for specific advice."
    }

@app.post("/predict")
async def predict_liquidity(request: PredictionRequest):
    model_key = next((k for k in MODELS.keys() if k.lower() == request.agent_id.lower()), None)
    # Fetch from RAM (Instantly)
    model = MODELS.get(model_key)

    if not model:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_id}' not found in system.")
    
    try:
        exog_input = [[
            request.is_holiday, 
            request.is_payday_window, 
            request.is_school_window, 
            request.is_weekend
        ]]

        forecast = model.get_forecast(steps=1, exog=exog_input)
        res = forecast.summary_frame()
        val = res['mean'].iloc[0]

        return {
            "agent_id": request.agent_id,
            "prediction_kes": round(val, 2),
            "safety_buffer_high": round(res['mean_ci_upper'].iloc[0], 2),
            "action": "WITHDRAW" if val > 0 else "DEPOSIT",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/network_summary")
async def get_all_forecasts():
    report = []
    for name, model in MODELS.items():
        try:
            # Base case: [0,0,0,0]
            f = model.get_forecast(steps=1, exog=[[0,0,0,0]])
            r = f.summary_frame()
            expected = r['mean'].iloc[0]
            
            report.append({
                "agent": name,
                "expected_need": round(expected, 2),
                "risk": "High" if r['mean_ci_upper'].iloc[0] > 100000 else "Normal"
            })
        except:
            continue
    return {"date": pd.Timestamp.now().date().isoformat(), "agents": report}

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)