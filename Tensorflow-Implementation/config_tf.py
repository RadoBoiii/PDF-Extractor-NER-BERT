"""
Access the config.py folder in the Tensorflow Implementation folder.
This file contains the configurations required excuting the BERT model using Tensorflow.
Any customizations that need to be made can be done by changing the file paths in the config.py
"""
import transformers
import os
import re
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
import warnings

warnings.filterwarnings("ignore")

MAX_LEN = 400
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 10
BASE_MODEL_PATH = r"../FLASK/Resource/bert_base_uncased"
MODEL_PATH = "../FLASK/Resource/my_model.h5"
TRAINING_FILE = "/Users/sudhirshinde/Downloads/input/df_final.csv"
#WEIGHT_PATH = "/content/drive/MyDrive/EdgeML_Team/Bert-Entity-Extraction using TensorFlow/saved_model/my_model_weights_1.h5"
EXTRACTED_FILE = "/Users/sudhirshinde/Desktop/PythonForEverybody/PDF-Extractor-NER-BERT/Tensorflow-Implementation/Results/extracted_info.csv" 
TEXT_FILE_PATH = r'/Users/sudhirshinde/Downloads/Text/*.txt'
# encoder = TFBertModel.from_pretrained("/content/drive/MyDrive/BERT/input/bert-base-uncased")
tokenizer = BertWordPieceTokenizer("../FLASK/Resource/bert_base_uncased/bert-base-uncased-vocab.txt", lowercase=True)
