import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random data
def generate_random_data(start_date, days):
    dates = [start_date + timedelta(days=x) for x in range(days)]
    energy_usage = np.random.uniform(100, 500, size=days)  # Example range
    water_usage = np.random.uniform(50, 300, size=days)    # Example range
    trash_amount = np.random.uniform(10, 100, size=days)   # Example range
    
    data = {
        'date': dates,
        'energy_usage': energy_usage,
        'water_usage': water_usage,
        'trash_amount': trash_amount
    }
    
    return pd.DataFrame(data)

# Generate the dataset
start_date = datetime(2023, 1, 1)
days = 500

df = generate_random_data(start_date, days)
df.to_csv('usage_data.csv', index=False)

print("CSV dataset generated.")
