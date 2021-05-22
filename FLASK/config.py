import warnings
from transformers import BertTokenizer, TFBertModel, BertConfig
from tokenizers import BertWordPieceTokenizer
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import string
import json
import re
import transformers
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings("ignore")

MAX_LEN = 400
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 10
BASE_MODEL_PATH = r"bert-base-uncased"
MODEL_PATH = r"/Users/iambankaratharva/CanspiritAI/bert-entity-extraction/FLASK/my_model.h5"
TRAINING_FILE = r"/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/input/df_final.csv"
#WEIGHT_PATH = r"models\my_model_weights_1.h5"
EXTRACTED_FILE = "extracted_info.csv"
TEXT_FILE_PATH = r'Text/*.txt'
# encoder = TFBertModel.from_pretrained("/content/drive/MyDrive/BERT/input/bert-base-uncased")
tokenizer = BertWordPieceTokenizer(
    r"/Users/iambankaratharva/CanspiritAI/bert-entity-extraction/FLASK/bert-base-uncased/vocab.txt", lowercase=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
