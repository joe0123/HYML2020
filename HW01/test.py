import sys
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

EPS = 1e-10

from data_preprocessing import test_data


if __name__ == "__main__":
    test_x = test_data(sys.argv[1])

# Predicting
    df = pd.DataFrame(columns=["id", "value"])
    df["id"] = ["id_{}".format(i) for i in range(test_x.shape[0])]
    if sys.argv[3] == "adagrad":
        test_x = np.concatenate((np.ones((test_x.shape[0], 1)).astype(float), test_x), axis=1)
        w = np.load("model.adagrad.npy")
        df["value"] = np.clip(np.dot(test_x, w), 0, np.inf)
    elif sys.argv[3] == "svr":
        import pickle
        with open("model.svr.pkl", "rb") as f:
            model = pickle.load(f)
        df["value"] = np.clip(model.predict(test_x), 0, np.inf)
    df.to_csv(sys.argv[2], index=False)
