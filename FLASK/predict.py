"""
Access this file from the FLASK folder.
This files predicts the outcome of the model. It consists of certain preprocessing steps which are fundamental in recieving clean and appropriate outcomes.  It converts the raw output model to a structured format.
"""
import warnings
import config
import pandas as pd
import glob
import joblib
import dataset
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings("ignore")


# In detail:-
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

def get_tokens(tokenized_sentence, tags, tags_name, enc_tag):
    """This function extracts tokens for the words in the sentences.

    :param tokenized_sentence: Recieves the sentences once it is tokenized, defaults to none
    
    :type data_path: list
    
    :param tags: It is the annotated tag that have to be predicted, defaults to none
    
    :type data_path: list
    
    :param tags_name: It is the tag name with the POS included, defaults to none
    
    :type data_path: list
    
    :param enc_tag: It recieves the encoded tag, defaults to none
    
    :type data_path: list
    
    :return: map_: returns encoded tokens of each word
    
    :rtype: dictionary
    """
    map_ = {}
    for i in enc_tag.classes_:
        map_[i] = []

    for i in range(len(tags_name)):
        if tags[i] != 16 and tokenized_sentence[i] != 101:
            map_[tags_name[i]].append(tokenized_sentence[i])
    return map_


def get_mapping(sentence, meta_file, model_bert):
    """Maps the words to the tokens and returns the predictions of each tag.

    :param sentence: Recieves the tokenized sentence, defaults to none
    
    :type data_path: list
    
    :param meta_file: Meta file from the config.py, defaults to none
    
    :type data_path: .bin file
    
    :param model_bert: NER BERT model from config.py, defaults to none
    
    :type data_path: .bin file
    
    :return: final_info: returns key information that must be extracted
    
    :rtype: dictionary
    """
    final_info = {"Account_no": [], "Admit_Date": [], "MR_Number": [], "Patient_Name": [],  "Social_Security_no": [], "Age": [
    ], "DOB": [], "Patient_Phone_no": [], "Admitting_Disease": [], "Admitting_Physician": [], "Primary_Insurance_Policy": []}
    extracted_info = {"Account_no": [], "Admit_Date": [], "MR_Number": [], "Patient_Name": [],  "Social_Security_no": [], "Age": [
    ], "DOB": [], "Patient_Phone_no": [], "Admitting_Disease": [], "Admitting_Physician": [], "Primary_Insurance_Policy": []}
    x_test, n_tokens = dataset.create_test_input_from_text(sentence)
    pred_test = model_bert.predict(x_test)
    pred_tags = np.argmax(pred_test, 2)[0][:n_tokens]
    tokenized_sentence = x_test[0][0][:n_tokens]
    meta_data = meta_file
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    le_dict = dict(zip(enc_tag.transform(
        enc_tag.classes_), enc_tag.classes_))
    tags_name = [le_dict.get(_, '[pad]') for _ in pred_tags]
    map = get_tokens(tokenized_sentence, pred_tags, tags_name, enc_tag)
    for i in map:
        # print(i,"-->",config.tokenizer.decode(map[i]))
        if i == "B-ACCOUNT #":
            extracted_info['Account_no'].append(
                config.tokenizer.decode(map[i]))
            final_info['Account_no'].append(
                extracted_info['Account_no'][0])

        if i == "B-ADMIT DATE":
            extracted_info['Admit_Date'].append(
                config.tokenizer.decode(map[i]))
            final_info['Admit_Date'].append(
                extracted_info['Admit_Date'][0])

        if i == "B-ADMITTING DIAGNOSIS":
            extracted_info['Admitting_Disease'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Admitting_Disease'][0] = extracted_info['Admitting_Disease'][0].split()[
                0]
        if i == "I-ADMITTING DIAGNOSIS":
            disease = config.tokenizer.decode(map[i])
            copy = config.tokenizer.decode(map[i])
            if len(copy.split()) > 1:
                dis = disease.split()[0]
                name = extracted_info['Admitting_Disease'][0] + ' ' + dis

            else:
                name = extracted_info['Admitting_Disease'][0] + \
                    ' ' + disease

            final_info['Admitting_Disease'].append(name)

        if i == "B-ADMITTING PHYSICIAN":
            extracted_info['Admitting_Physician'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Admitting_Physician'][0] = extracted_info['Admitting_Physician'][0].split()[
                0]
        if i == "I-ADMITTING PHYSICIAN":
            last_name = config.tokenizer.decode(map[i]).split()
            extracted_info['Admitting_Physician'].append(last_name[0])
            extracted_info['Admitting_Physician'][0] = ' '.join(
                extracted_info['Admitting_Physician'])
            extracted_info['Admitting_Physician'].pop()
            final_info['Admitting_Physician'].append(
                extracted_info['Admitting_Physician'][0])

        if i == "B-AGE":
            extracted_info['Age'].append(config.tokenizer.decode(map[i]))
            final_info['Age'].append(extracted_info['Age'][0])

        if i == "B-BIRTH DATE":
            extracted_info['DOB'].append(config.tokenizer.decode(map[i]))
            final_info['DOB'].append(extracted_info['DOB'][0])

        if i == "B-MR NUMBER":
            extracted_info['MR_Number'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['MR_Number'][0] = extracted_info['MR_Number'][0].replace(
                " ", "")
            extracted_info['MR_Number'][0] = extracted_info['MR_Number'][0][:8]
            final_info['MR_Number'].append(extracted_info['MR_Number'][0])

        if i == "B-PATIENT NAME":
            # print(config.tokenizer.decode(map[i]))
            extracted_info['Patient_Name'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Patient_Name'][0] = extracted_info['Patient_Name'][0].split()[
                0]
        if i == "I-PATIENT NAME":
            extracted_info['Patient_Name'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Patient_Name'][0] = ' '.join(
                extracted_info['Patient_Name'])
            extracted_info['Patient_Name'].pop()
            final_info['Patient_Name'].append(
                extracted_info['Patient_Name'][0])

        if i == "B-PATIENT PHONE #":
            extracted_info['Patient_Phone_no'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0].replace(
                "(", "")
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0].replace(
                ")", "")
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0].replace(
                ".", "")
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0].replace(
                " ", "")
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0].replace(
                "-", "")
            extracted_info['Patient_Phone_no'][0] = extracted_info['Patient_Phone_no'][0][:10]
            final_info['Patient_Phone_no'].append(
                extracted_info['Patient_Phone_no'][0])

        if i == "B-PRIMARY INSURANCE PLAN":
            ip = config.tokenizer.decode(map[i])
            # print(ip)
            # extracted_info['Primary_Insurance_Policy'].append(config.tokenizer.decode(map[i]))
            if ip == "":
                extracted_info['Primary_Insurance_Policy'].append("-")
            else:
                extracted_info['Primary_Insurance_Policy'].append(ip)
            extracted_info['Primary_Insurance_Policy'][0] = extracted_info['Primary_Insurance_Policy'][0].split()[
                0]
        if i == "I-PRIMARY INSURANCE PLAN":
            iip = config.tokenizer.decode(map[i])
            # print(iip)
            # extracted_info['Primary_Insurance_Policy'].append(config.tokenizer.decode(map[i]))
            if iip == "":
                # print("inside")
                extracted_info['Primary_Insurance_Policy'].append("care")
            else:
                extracted_info['Primary_Insurance_Policy'].append(iip)
            care = extracted_info['Primary_Insurance_Policy'][1].split()[0]
            if extracted_info['Primary_Insurance_Policy'][0] == "-":
                care_name = "-"
            else:
                care_name = extracted_info['Primary_Insurance_Policy'][0] + ' ' + care
            # extracted_info['Primary_Insurance_Policy'].pop()
            final_info['Primary_Insurance_Policy'].append(care_name)

        if i == "B-SOCIAL SECURITY #":
            extracted_info['Social_Security_no'].append(
                config.tokenizer.decode(map[i]))
        if i == "I-SOCIAL SECURITY #":
            extracted_info['Social_Security_no'].append(
                config.tokenizer.decode(map[i]))
            extracted_info['Social_Security_no'][0] = ' '.join(
                extracted_info['Social_Security_no'])
            extracted_info['Social_Security_no'].pop()

            extracted_info['Social_Security_no'][0] = extracted_info['Social_Security_no'][0].replace(
                " ", "")
            extracted_info['Social_Security_no'][0] = extracted_info['Social_Security_no'][0].replace(
                "-", "")
            extracted_info['Social_Security_no'][0] = extracted_info['Social_Security_no'][0].replace(
                "(", "")
            extracted_info['Social_Security_no'][0] = extracted_info['Social_Security_no'][0].replace(
                ")", "")
            extracted_info['Social_Security_no'][0] = extracted_info['Social_Security_no'][0][:9]
            #extracted_info['Social_Security_no'][0] = int(extracted_info['Social_Security_no'][0])
            final_info['Social_Security_no'].append(
                extracted_info['Social_Security_no'][0])
    return final_info


# if __name__ == "__main__":
#     meta_data = joblib.load("meta.bin")
#     enc_tag = meta_data["enc_tag"]
#     num_tag = len(list(enc_tag.classes_))
#     m1 = create_model(num_tag)
#     m1.load_weights(config.WEIGHT_PATH)

#     TEXT_FILE_PATH = r'Text/*.txt'
#     var = glob.glob(TEXT_FILE_PATH)
#     var.sort()
#     for i in var:
#         f = open(i, "r")

#         sentence = f.read()

#         get_mapping([sentence])

#     extracted_df = pd.DataFrame(final_info)
#     print(extracted_df)
#     extracted_df.to_csv(config.EXTRACTED_FILE, header=True, index=False)
