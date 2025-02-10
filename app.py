from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('model/heart_disease_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    age = int(request.form['age'])
    cp = int(request.form['cp'])
    thalach = int(request.form['thalach'])

    # Prepare data for prediction
    input_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


