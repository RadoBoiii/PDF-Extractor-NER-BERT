"""
Access the model.py folder in Tensorflow Implementation folder.
The model.py file has 2 important functions that are pertinent to creation of the model.
"""
import config
import tensorflow as tf
import transformers
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig

 

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction=tf.keras.losses.Reduction.NONE
)

def masked_ce_loss(real, pred):
    """This function calculates the masked loss to reduce the effect of loss due to padding on the model.

    :param real: Actual Label in the processed data, defaults to none
    
    :type data_path: list
    
    :param pred: Predicited Label, defaults to none
    
    :type data_path: list
    
    :return: loss_: returns the mean loss
    
    :rtype: float
    """
    mask = tf.math.logical_not(tf.math.equal(real, 17))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def create_model(num_tags):
    """This function calculates the masked loss to reduce the effect of loss due to padding on the model.

    :param num_tags: the total number of unique tags, defaults to none
    
    :type data_path: int
    
    :return: model: returns the BERT model
    
    :rtype: tensorflow model
    """
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(config.BASE_MODEL_PATH)
    # encoder =  config.encoder
    ## NER Model
    input_ids = layers.Input(shape=(config.MAX_LEN,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(config.MAX_LEN,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(config.MAX_LEN,), dtype=tf.int32)
    embedding = encoder.bert(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    embedding = layers.Dropout(0.3)(embedding)
    tag_logits = layers.Dense(num_tags+1, activation='softmax')(embedding)
    
    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(lr=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model