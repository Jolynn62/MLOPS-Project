from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and preprocessing pipeline
with open('model/mushroom_pipeline.pkl', 'rb') as file:
    mushroom_pipeline = pickle.load(file)

# Load the label encoders
with open('model/label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')  # HTML file for the form

@app.route('/predict', methods=['POST'])
def predict():
    # Get only selected features' input data from the form
    input_data = {
        'cap-shape': request.form.get('cap-shape'),
        'cap-color': request.form.get('cap-color'),
        'odor': request.form.get('odor'),
        'habitat': request.form.get('habitat')
    }

    # Set default values for the unselected features
    default_values = {
        'cap-surface': 'smooth',
        'bruises': 'no',
        'gill-attachment': 'free',
        'gill-spacing': 'close',
        'gill-size': 'broad',
        'gill-color': 'black',
        'stalk-shape': 'tapering',
        'stalk-root': 'rooted',
        'stalk-surface-above-ring': 'smooth',
        'stalk-surface-below-ring': 'smooth',
        'stalk-color-above-ring': 'white',
        'stalk-color-below-ring': 'white',
        'veil-type': 'partial',
        'veil-color': 'white',
        'ring-number': 'one',
        'ring-type': 'evanescent',
        'spore-print-color': 'black',
        'population': 'scattered'
    }

    # Convert input into a dataframe
    input_df = pd.DataFrame([input_data])

    # Add default values for unselected features
    for feature in default_values:
        if feature not in input_df.columns:
            input_df[feature] = default_values[feature]

    # Apply label encoding
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Apply preprocessing pipeline and make predictions
    prediction = mushroom_pipeline.predict(input_df)

    # Return prediction result
    result = 'Poisonous' if prediction[0] == 1 else 'Edible'
    
    # Render a template with the result
    return render_template('result.html', prediction_text=f'The mushroom is {result}')

if __name__ == '__main__':
    app.run(debug=True)