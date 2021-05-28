"""
Access the train.py folder in Tensorflow Implementation folder.
The train.py file is one of the most important files in this Project. The steps performed in this file are:

#. It accesses the 3 parameters that are returned from dataset.py file and stores them in the meta.bin file.

#. The model architecture is initialized to be trained on a TPU/GPU automatically.

#. Model is fit and then the model metrics are visualized. The metrics include:
  * Accuracy
  * Loss

4. The entire model is saved after training.
"""

import joblib
import dataset
import config 
import numpy as np
import tensorflow as tf
import pandas as pd
import model
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

class Metrics(Callback):
    """This Class serves as callbacks for calculating metrics such as precision, recall, accuracy and F1 score.
    """
    def __init__(self, validation_data, val_targ):
        self.validation_data = validation_data
        self.val_targ = val_targ

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        predicted = np.asarray(self.model.predict(self.validation_data))
        val_predict = np.array([i.argmax(axis = 1) for i in predicted])
        m = MultiLabelBinarizer().fit(self.val_targ)
        val_targ=m.transform(self.val_targ)
        val_predict= m.transform(val_predict)
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f" — val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}")
        return




if __name__ == '__main__':
    print("=======================================1=======================================")
    x_train, y_train, tag_encoder = dataset.create_inputs_targets(config.TRAINING_FILE)
    X_train_0,X_test_0,X_train_1,X_test_1, X_train_2, X_test_2, Y_train, Y_test = train_test_split(x_train[0], x_train[1],x_train[2],y_train, test_size=0.25, random_state=42)
    split_x_train = [X_train_0,X_train_1,X_train_2]
    split_x_test = [X_test_0,X_test_1,X_test_2]
    print("=======================================2=======================================")
    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]
    num_tags = len(list(enc_tag.classes_))
    print("=======================================3=======================================")
  

    use_tpu = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        use_tpu = True
    except:
        use_tpu = False

    if use_tpu:
        # Create distribution strategy
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)

        # Create model
        with strategy.scope():
            my_model = model.create_model(num_tags)
    else:
        my_model = model.create_model(num_tags)
    
    print(my_model.summary())

    print("=======================================4=======================================")
    metrics = Metrics(split_x_test,Y_test)
    bs = 64 if use_tpu else 16
    print("=======================================5=======================================")
    history = my_model.fit(
        split_x_train,
        Y_train,
        validation_data=(split_x_test, Y_test),
        epochs=5,
        verbose=1,
        callbacks = [metrics],
        batch_size=64)
    print("=======================================6=======================================")

    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()
    my_model.save(config.MODEL_PATH)
    # model.save_weights(config.WEIGHT_PATH)
    
    
