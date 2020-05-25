import sys  # python train_lda.py [x] [y] [b]
import numpy as np
np.random.seed(1114)
np.set_printoptions(threshold=np.inf)

from data_preprocessing import train_data
from functions import f, accuracy


if __name__ == "__main__":
# Data Preprocessing
    _, train_x, train_y, _ = train_data(sys.argv[1], sys.argv[2], sys.argv[3], val_split=0)
    
# Computing
    num = np.zeros((2))
    mean = np.zeros((2, train_x.shape[-1]))
    cov = np.zeros((2, train_x.shape[-1], train_x.shape[-1]))
    for i in range(2):
        target = np.where(train_y == i)[0]
        num[i] = target.shape[0]
        mean[i] = (np.sum(train_x[target], axis=0) / num[i])
        cov[i] = np.dot((train_x[target] - mean[i]).T, (train_x[target] - mean[i]))
    cov = np.sum(cov, axis=0) / train_x.shape[0]
    # Because cov might be singular, so np.linalg.inv() may cause a large numerical error
    cov_inv = np.linalg.pinv(cov)

    w = np.dot(cov_inv.T, mean[1] - mean[0])
    b = -0.5 * np.dot(np.dot(mean[1].T, cov_inv), mean[1]) + 0.5 * np.dot(np.dot(mean[0].T, cov_inv), mean[0]) + np.log(num[1] / num[0])
    w = np.vstack((b.reshape((1, 1)), w.reshape(w.shape[0], 1)))
    np.save("model.lda", w)

# Evaluating
    train_x = np.concatenate((np.ones((train_x.shape[0], 1)).astype(float), train_x), axis=1)
    train_y = train_y.reshape(train_y.shape[0], 1)

    pred_y = f(w, train_x)
    acc = np.around(accuracy(np.around(pred_y), train_y), decimals=4)
    print(acc)
