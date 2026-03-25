import pandas as pd
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import numpy as np

#paths
DATA_PATH = r"C:\Users\Jones Mbela\Desktop\RENNY\AI AND ML\MPESA Liquidity\data\synthetic_mobile_money_data.csv"
MODEL_PATH = r"C:\Users\Jones Mbela\Desktop\RENNY\AI AND ML\MPESA Liquidity\Model"

#Agents that need training
remaining_agents = [
    "Agent_Rural_Med", 
    "Agent_Rural_High", 
    "Agent_Rural_Low", 
    "Agent_Urban_Med"
]

# Training Batch Function
def train_batch():

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)

    results_report= []

    for agent_id in remaining_agents:
        print(f"----Processing {agent_id}----")

        #Filtering spesific agent
        agent_df = df[df['agent_id'] == agent_id].asfreq('D').fillna(0)

        agent_df['is_weekend'] = agent_df['day_of_week'].isin([4, 5, 6]).astype(int)
        #target
        y= agent_df['net_liquidity_needed']

        exog = agent_df[['is_holiday', 'is_payday_window', 'is_school_window', 'is_weekend']].astype(float)
        

        #validation step
        train_y,test_y = y.iloc[:-30],y.iloc[-30:]
        train_exog,test_exog = exog.iloc[:-30],exog.iloc[-30:]



        #SARIMAX MODEL
        val_model = SARIMAX(
            train_y,
            train_exog,
            order=(1,0,1),
            seasonal_order = (1,0,1,7),
            enforce_stationarity= False
        )

        print(f"Fitting model for {agent_id}....")
        val_results = val_model.fit(disp=False)
        #forcasting and errors
        forecast = val_results.get_forecast(steps=30,exog=test_exog)
        mae = mean_absolute_error (test_y,forecast.summary_frame()['mean'])

        print (f"Standard error (MAE) for {agent_id}: {mae:,.2f} KES ")

        #final model training
        final_model = SARIMAX(
            y,
            exog= exog,
            order = (1,0,1),
            seasonal_order= (1,0,1,7),
            enforce_stationarity= False

        )
        final_results = final_model.fit(disp=False)

        clean_name = str(agent_id).strip()
        file_name = f"sarimax_{clean_name}.pkl"


        #saving model
        save_path = os.path.join(MODEL_PATH, file_name)
        joblib.dump(final_results,save_path)

        results_report.append({"Agent":agent_id,"MAE":mae})

        print(f"Saved to: {save_path}")
    #Print summary table
    print("\n" + "="*30)
    print("FINAL BATCH TRAINING REPORT")
    print("="*30)
    report_df = pd.DataFrame(results_report)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    train_batch()
    print("Batch training complete for all remaining agents")