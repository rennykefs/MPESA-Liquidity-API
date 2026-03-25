import pandas as pd
import numpy as np
from datetime import datetime

def generate_agent_data(agent_id, start_date, end_date, base_volume, agent_type='urban'):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    # 1. Base Volume and random noise (.1)
    transaction_volume = np.random.normal(base_volume, base_volume*0.1, n)
    transaction_volume = np.maximum(transaction_volume, base_volume*0.1)

    # Use a single split factor to ensure cash_in + cash_out = total volume
    split_factor = np.random.uniform(0.45, 0.55, n)
    
    df = pd.DataFrame({
        'date': dates,
        'agent_id': agent_id,
        'cash_in': transaction_volume * split_factor,
        'cash_out': transaction_volume * (1 - split_factor)
    })

    # 2. Date feature engineering
    df = df.set_index('date')
    df['day_of_week'] = df.index.dayofweek
    df['day'] = df.index.day
    df['month'] = df.index.month

    # 3. Define Boolean Windows for Events
    # Payday
    df['is_payday_window'] = (df['day'] >= 26) | (df['day'] <= 3)
    
    # School fees
    df['is_school_window'] = (df['month'].isin([1, 5, 9])) & (df['day'] <= 10)
    
    # Holidays
    is_christmas = (df['month'] == 12) & (df['day'] >= 20) | (df['month'] == 1) & (df['day'] <= 3)
    easter_dates = df.index.isin(['2022-04-15', '2022-04-16', '2022-04-17', '2022-04-18',
                                  '2023-04-07', '2023-04-08', '2023-04-09', '2023-04-10',
                                  '2024-03-29', '2024-03-30', '2024-03-31', '2024-04-01'])
    eid_dates = df.index.isin(['2022-05-01', '2022-05-02', '2022-05-03',
                               '2023-04-20', '2023-04-21', '2023-04-22',
                               '2024-04-09', '2024-04-10', '2024-04-11'])
    df['is_holiday'] = is_christmas | easter_dates | eid_dates

    #   HIERARCHY 
    # Order: Holiday (highest) -> School Fees -> Payday
    conditions = [
        (df['is_holiday']),
        (df['is_school_window']),
        (df['is_payday_window'])
    ]

    if agent_type == 'urban':
        out_choices = [1.2, 1.4, 1.2]
        in_choices  = [2.5, 0.8, 1.3]
    else: # rural
        out_choices = [2.8, 1.15, 1.5]
        in_choices  = [1.1, 0.9, 1.1]

    df['mult_out'] = np.select(conditions, out_choices, default=1.0)
    df['mult_in']  = np.select(conditions, in_choices, default=1.0)

    # Apply event multipliers
    df['cash_out'] *= df['mult_out']
    df['cash_in']  *= df['mult_in']

    # 5. Day of the week effect (Applied after event multipliers)
    # Mon=0, Tue=1, ..., Fri=4, Sat=5, Sun=6
    day_multipliers = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.1])
    df['cash_in']  *= day_multipliers[df['day_of_week']]
    df['cash_out'] *= day_multipliers[df['day_of_week']]

    # 6. Macro economic factors (Applied as a baseline shift)
    inflation_mask = (df.index >= '2023-06-01') & (df.index <= '2024-03-31')
    df['cash_out'] = np.where(inflation_mask, df['cash_out'] * 1.10, df['cash_out'])
    df['cash_in']  = np.where(inflation_mask, df['cash_in'] * 0.95, df['cash_in'])

    # 7. Final Liquidity Calculation
    df['net_liquidity_needed'] = df['cash_out'] - df['cash_in']
    
    return df

# Agents Definition
agents = [
    ('Agent_Urban_High', 500000, 'urban'),
    ('Agent_Urban_Med', 300000, 'urban'),
    ('Agent_Rural_High', 200000, 'rural'),
    ('Agent_Rural_Med', 100000, 'rural'),
    ('Agent_Rural_Low', 50000, 'rural')
]

# Run Generator
full_data = pd.concat([generate_agent_data(a[0],'2022-01-01', '2024-12-31', a[1], a[2]) for a in agents])

# Save to CSV using Raw String for Windows Paths
full_data.to_csv(r"C:\Users\Jones Mbela\Desktop\RENNY\AI AND ML\MPESA Liquidity\data\synthetic_mobile_money_data.csv")

print("Phase 1 data generation complete. Dataset saved.")