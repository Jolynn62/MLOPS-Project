from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('model/hdb_resale_pipeline')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        'block': request.form['block'],
        'street_name': request.form['street_name'],
        'town': request.form['town'],
        'postal_code': request.form['postal_code'],
        'transaction_month': request.form['transaction_month'],
        'flat_type': request.form['flat_type'],
        'storey_range': request.form['storey_range'],
        'floor_area_sqm': float(request.form['floor_area_sqm']),
        'flat_model': request.form['flat_model'],
        'lease_start_year': int(request.form['lease_start_year']),
        'latitude': float(request.form['latitude']),
        'longitude': float(request.form['longitude']),
        'distance_to_cbd': float(request.form['distance_to_cbd']),
        'distance_to_mrt': float(request.form['distance_to_mrt'])
    }
    data['transaction_month'] = pd.to_datetime(data['transaction_month'], format='%Y-%m')
    # Convert data into a DataFrame
    data_df = pd.DataFrame([data])
    
    # Generate predictions
    predictions = predict_model(model, data=data_df)

    # Return the result
    result = predictions['prediction_label'][0]
    return render_template('index.html', prediction_text=f'Estimated Resale Price: SGD {result:,.2f}')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction['prediction_label'][0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True, port=5001)

