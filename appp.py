from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'Predicted Price: ₹{int(prediction[0]):,}')

if __name__ == '__main__':
    app.run(debug=True)
