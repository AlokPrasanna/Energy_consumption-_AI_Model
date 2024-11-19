import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from datetime import datetime

# Load the dataset
df = pd.read_csv('usage_data.csv')

# Convert date to ordinal for regression
df['date'] = pd.to_datetime(df['date']).map(datetime.toordinal)

# Split the data into features and targets
X = df[['date']]
y_energy = df['energy_usage']
y_water = df['water_usage']
y_trash = df['trash_amount']

# Split into train and test sets
X_train, X_test, y_train_energy, y_test_energy = train_test_split(X, y_energy, test_size=0.2, random_state=42)
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X, y_water, test_size=0.2, random_state=42)
X_train_trash, X_test_trash, y_train_trash, y_test_trash = train_test_split(X, y_trash, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the models
model_energy = LinearRegression()
model_water = LinearRegression()
model_trash = LinearRegression()

model_energy.fit(X_train_scaled, y_train_energy)
model_water.fit(X_train_scaled, y_train_water)
model_trash.fit(X_train_scaled, y_train_trash)

# Evaluate the models
y_pred_energy = model_energy.predict(X_test_scaled)
y_pred_water = model_water.predict(X_test_scaled)
y_pred_trash = model_trash.predict(X_test_scaled)

print(f"Energy Usage MSE: {mean_squared_error(y_test_energy, y_pred_energy)}")
print(f"Water Usage MSE: {mean_squared_error(y_test_water, y_pred_water)}")
print(f"Trash Amount MSE: {mean_squared_error(y_test_trash, y_pred_trash)}")

# Save the trained models to files
joblib.dump(model_energy, 'model_energy.pkl')
joblib.dump(model_water, 'model_water.pkl')
joblib.dump(model_trash, 'model_trash.pkl')

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved to disk.")
