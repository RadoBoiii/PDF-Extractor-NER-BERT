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
BASE_MODEL_PATH = r"/content/drive/MyDrive/BERT/input/bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/input/df_final.csv"
WEIGHT_PATH = "/content/drive/MyDrive/EdgeML_Team/Bert-Entity-Extraction using TensorFlow/saved_model/my_model_weights_1.h5"
EXTRACTED_FILE = "/content/drive/MyDrive/EdgeML_Team/Bert-Entity-Extraction using TensorFlow/Results/extracted_info.csv" 
TEXT_FILE_PATH = r'/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/Dataset/Text /*.txt'
# encoder = TFBertModel.from_pretrained("/content/drive/MyDrive/BERT/input/bert-base-uncased")
tokenizer = BertWordPieceTokenizer("/content/drive/MyDrive/BERT/input/bert-base-uncased/vocab.txt", lowercase=True)
