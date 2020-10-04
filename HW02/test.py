import sys  # python test.py [model] [x] [b] [o]
import numpy as np
import pandas as pd

from data_preprocessing import test_data



if __name__ == "__main__":
    col_name, test_x = test_data(sys.argv[2], sys.argv[3])
    df = pd.DataFrame(columns=["id", "label"])
    df["id"] = range(test_x.shape[0])

    if sys.argv[1] in ["logi", "lda"]:
        from functions import f
        test_x = np.concatenate((np.ones((test_x.shape[0], 1)).astype(float), test_x), axis=1)
        w = np.load("model.{}.npy".format(sys.argv[1]))
        #print([i for i in range(w.shape[0] - 1) if w[i + 1] < 0.02])
        df["label"] = np.around(f(w, test_x)).astype(int)
    elif sys.argv[1] == "dnn":
        import os
        import tensorflow as tf
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*1)])
        from keras.models import load_model

        model = load_model("model.dnn.hdf5")
        df["label"] = np.around(model.predict(test_x)).astype(int)

    df.to_csv(sys.argv[4], index=False)
