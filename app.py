# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your saved model
model = joblib.load('housing_model.pkl')

# 1. Serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

# 2. Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the website's form
    data = [float(x) for x in request.form.values()]
    
    # Convert it into the 2D array format the model expects
    final_features = [np.array(data)]
    
    # Make the prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    # Send the result back to the website
    return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)