import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

MAX_LEN = 300
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 20

BASE_MODEL_PATH = "/content/drive/MyDrive/BERT/input/bert-base-uncased"
MODEL_PATH = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/Models/model_atharva.bin"
META_FILE = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/Models/meta_atharva.bin"
TRAINING_FILE = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/input/df_final.csv"
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
EXTRACTED_FILE = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/Results/extracted_results_frompyfile.csv"
TEXT_FILE_PATH = "/content/drive/MyDrive/EdgeML_Team/bert-entity-extraction/Dataset/Text /*.txt"