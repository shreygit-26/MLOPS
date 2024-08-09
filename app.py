import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

file = pickle.load(open('Modify.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_defe():
    dde = float(request.form.get('DE'))
    fee = float(request.form.get('FE'))

    # prediction
    result = file.predict(np.array([dde,fee]).reshape(1, 2))


    if result[0] >0:
        result = '1'
    else:
        result = '0'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8685)
