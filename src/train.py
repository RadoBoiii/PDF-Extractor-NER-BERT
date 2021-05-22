import joblib
import dataset
import config 
import numpy as np
import tensorflow as tf
import pandas as pd
import model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print("=======================================1=======================================")
    x_train, y_train, tag_encoder = dataset.create_inputs_targets(config.TRAINING_FILE)
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
    
    my_model.summary()
    print("=======================================4=======================================")
    
    bs = 64 if use_tpu else 16
    print("=======================================5=======================================")
    history = my_model.fit(
        x_train,
        y_train,
        epochs=config.EPOCHS,
        verbose=1,
        batch_size=bs,
        validation_split=0.1)
    print(1)

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
    
    my_model.save_weights(config.WEIGHT_PATH)