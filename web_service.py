from flask import Flask, request, jsonify
import joblib
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load the models and scaler
model_energy = joblib.load('model_energy.pkl')
model_water = joblib.load('model_water.pkl')
model_trash = joblib.load('model_trash.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['GET'])
def predict_usage():
    # Get date from request
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({"error": "No date provided"}), 400
    
    try:
        date_ordinal = datetime.strptime(date_str, '%Y-%m-%d').toordinal()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use yyyy-MM-dd."}), 400
    
    # Prepare the data for prediction
    X_new = np.array([[date_ordinal]])
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    energy_pred = model_energy.predict(X_new_scaled)
    water_pred = model_water.predict(X_new_scaled)
    trash_pred = model_trash.predict(X_new_scaled)
    
    # Return the predictions as JSON
    return jsonify({
        'date': date_str,
        'predicted_energy_usage': energy_pred[0],
        'predicted_water_usage': water_pred[0],
        'predicted_trash_amount': trash_pred[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
