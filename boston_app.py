import numpy as np
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('boston_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])
    prediction = model.predict(final_input)
    float_prediction=float(prediction[0])
    return render_template('boston_index.html', prediction_text=f'Predicted House Price: ${float_prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)

