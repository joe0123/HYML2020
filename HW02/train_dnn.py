import sys  # python train_dnn.py [x] [y] [b]
seed = 1114
import numpy as np
np.random.seed(seed)
np.set_printoptions(threshold=np.inf)
import os
os.environ['PYTHONHASHSEED']=str(seed)
import random
random.seed(seed)
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*1)])
tf.random.set_seed(seed)
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks.callbacks import ModelCheckpoint

from data_preprocessing import train_data

BATCH = 32
EPOCH = 30
LR = 1e-4
EPS = 1e-10


if __name__ == "__main__":
# Data Preprocessing
    _, train_x, train_y, val_data = train_data(sys.argv[1], sys.argv[2], sys.argv[3], val_split=0.2)

# Model Construction and Training 
    feature_input = Input(shape=(train_x.shape[-1], ))
    nn = Dense(64, activation="relu")(feature_input)
    nn = Dropout(0.5)(nn)
    nn = Dense(128, activation="relu")(nn)
    nn = Dropout(0.5)(nn)
    output = Dense(1, activation="sigmoid")(nn)

    model = Model(feature_input, output)
    print(model.summary())
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR), metrics=["acc"])
    
    callbacks = [#EarlyStopping(monitor="val_loss", patience=15), 
            ModelCheckpoint("model.dnn.hdf5", monitor="val_acc", verbose=1, save_best_only=True, mode="max")]
    model.fit(train_x, train_y, batch_size=BATCH, epochs=EPOCH, validation_data=val_data, callbacks=callbacks, class_weight={0: 1, 1: 1.2})
    #model.save("model.dnn.hdf5")
