import joblib
import pandas as pd
import os

MODEL_FOLDER = r"C:\Users\Jones Mbela\Desktop\RENNY\AI AND ML\MPESA Liquidity\Model"
agent_to_test = "Agent_Rural_High"
model_path = os.path.join(MODEL_FOLDER, f"sarimax_{agent_to_test}.pkl")

print(f"Loading model for {agent_to_test}....")
loaded_results = joblib.load(model_path)

#Defining scenario for tomorrow example payday but not a holiday or weekend
# observing model order
tomorrow_hints = [[0,1,0,0]]

#forecast
forecast = loaded_results.get_forecast(steps=1,exog=tomorrow_hints)
prediction = forecast.summary_frame()

#result
predicted_val = prediction['mean'].iloc[0]
lower_bound = prediction['mean_ci_lower'].iloc[0]
upper_bound = prediction['mean_ci_upper'].iloc[0]

print("-" * 30)
print(f"RESULTS FOR {agent_to_test}:")
print(f"Predicted Net Liquidity: {predicted_val:,.2f} KES")
print(f"95% Confidence Range: {lower_bound:,.0f} to {upper_bound:,.0f} KES")
print("-" * 30)

if predicted_val > 0:
    print(f"ADVICE: Tell the agent to WITHDRAW cash from the bank.")
else:
    print(f"ADVICE: Tell the agent to DEPOSIT excess cash at the bank.")