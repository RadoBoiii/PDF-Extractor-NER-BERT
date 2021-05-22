from flask import Flask, render_template, request
import joblib
import config
import predict
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf

app = Flask(__name__)

meta_data = joblib.load(
    "/Users/iambankaratharva/CanspiritAI/bert-entity-extraction/FLASK/meta.bin")

model_bert = tf.keras.models.load_model(
    '/Users/iambankaratharva/CanspiritAI/bert-entity-extraction/FLASK/my_model.h5', compile=False)


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_():
    if request.method == 'POST':
        f = request.files['file']
        sentence = f.read().decode("utf-8")
        final_info = predict.get_mapping([sentence], meta_data, model_bert)
        return final_info


if __name__ == '__main__':
    app.run(debug=True)
