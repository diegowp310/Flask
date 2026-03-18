import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    output = int(prediction[0])

    if output == 1:
        result = 'This passenger would have SURVIVED'
    else:
        result = 'This passenger would NOT have survived'

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
