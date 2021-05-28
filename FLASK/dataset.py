"""
Access the dataset.py folder in Tensorflow Implementation folder.
This file contains functions for processing the csv files and creating input tokens for train set and test set.
No customizations are required in this file while using BERT model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
import tensorflow as tf
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
import config
import joblib
import numpy as np 
from sklearn import preprocessing
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
 


def process_csv(data_path):
    """This function performs label encoding on the columns and groups the rows from the data by sentence number.
    It also fill in the null values using 'forward fill'

    :param data_path: It is the absolute path to the csv file, defaults to none
    
    :type data_path: csv
    
    :return: Returns 3 paramters:
    
            - sentences: Sentences extracted from the csv file
            
            - tag: Annotated tag of key words 
            
            - enc_tag: encoded tag of each word by BERT
    
    :rtype: list
    """
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    enc_tag = preprocessing.LabelEncoder()
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag


def create_inputs_targets(data_csv):
    """This function tokenizes and encodes the tags of the training and performs masking and padding on the data if required.

    :param data_csv: The csv file is passed to this function, defaults to none
    
    :type data_path: csv
    
    :return: Returns 3 paramters:
    
             - x: a list which consists of input ids, token type ids, attention masks
            
             - y: list of tags
            
             - tag_encoder: the encoded tags
    
    :rtype: list
    """
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "tags": []
    }
    sentences, tags, tag_encoder = process_csv(data_csv)
    meta_data = {
        "enc_tag": tag_encoder
    }
    joblib.dump(meta_data, "meta.bin")
    
    for sentence, tag in zip(sentences, tags):
        input_ids = []
        target_tags = []
        for idx, word in enumerate(sentence):
            ids = config.tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
            num_tokens = len(ids)
            target_tags.extend([tag[idx]] * num_tokens)
        
        
        # Pad truncate
        input_ids = input_ids[:config.MAX_LEN - 2]
        target_tags = target_tags[:config.MAX_LEN - 2]

        input_ids = [101] + input_ids + [102]
        target_tags = [16] + target_tags + [16]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_len = config.MAX_LEN - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tags = target_tags + ([17] * padding_len)
        
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["tags"].append(target_tags)
        assert len(target_tags) == config.MAX_LEN, f'{len(input_ids)}, {len(target_tags)}'
        
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["tags"]
    return x, y, tag_encoder    

def create_test_input_from_text(texts):
    """This function tokenizes and encodes the tags of the testing and performs masking and padding on the data if required.


    :param data_csv: The csv file is passed to this function, defaults to none
    
    :type data_path: csv
    
    :return: Returns 2 paramters:
    
            - x: a list which consists of input ids, token type ids, attention masks
            
            - n_tokens: Length of input ids
    
    :rtype: list
    """
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    for sentence in texts:
        input_ids = []
        for idx, word in enumerate(sentence.split()):
            ids = config.tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
            num_tokens = len(ids)
            
        # Pad and create attention masks.
        # Skip if truncation is needed
        input_ids = input_ids[:config.MAX_LEN - 2]

        input_ids = [101] + input_ids + [102]
        n_tokens = len(input_ids)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_len = config.MAX_LEN - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    return x, n_tokens