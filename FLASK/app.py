"""
This is the main file to initialize flask and render html files through flask. It is the master file and calls all other secondary file to perform the extraction function.
"""

from flask import Flask, render_template, request, send_from_directory, current_app as app
import os
import pandas as pd
import shutil
import config
import predict
import joblib
import transformers
import tensorflow as tf
import pdf_to_img
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

meta_data = joblib.load(config.META_MODEL_PATH)

model_bert = tf.keras.models.load_model(config.MODEL_PATH, compile=False, custom_objects={
                                        'TFBertMainLayer': transformers.TFBertModel})

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def main():
    """Loads the main paage - Index.html
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Helps upload the pdf to the OCR functions. It ensures that the pdf is passed to image conversion function and then the latter  is passed through the OCR extractor.
    """
    global result
    pdf_target = os.path.join(APP_ROOT, 'static/pdf')
    img_target = os.path.join(APP_ROOT, 'static/pdf-images')

    # Preparing directory - pdf
    if not os.path.isdir(pdf_target):
        os.mkdir(pdf_target)

    # Preparing directory - image
    if not os.path.isdir(img_target):
        os.mkdir(img_target)

    # Uploading File
    for file in request.files.getlist('file'):
        filename = file.filename
        destination = "/".join([pdf_target, filename])
        file.save(destination)
        sentence = pdf_to_img.extractor_pytess(destination)
        final_info = predict.get_mapping([sentence], meta_data, model_bert)

    # Delete file
    if os.path.isdir(pdf_target):
        shutil.rmtree(pdf_target)

    if os.path.isdir(img_target):
        shutil.rmtree(img_target)

    #final_df = pd.DataFrame.from_dict(final_info,)
    # final_info.to_html('result.html')

    result = final_info
    return render_template('result.html', result=final_info)


@app.route('/download', methods=['POST'])
def download():
    """Displays the key-values successfully and allows download of the file in excel format. Returns the message 'Downloaded Succesfully' on downloading the file.
    """
    if request.method == 'POST':
        final_df = pd.DataFrame.from_dict(result)
        final_df.to_excel('result.xlsx', index=False)
    return 'Downloaded successfully!'


if __name__ == '__main__':
    app.run(debug=True)
