import sys  # python train_logi.py [x] [y] [b]
import numpy as np
np.random.seed(1114)
np.set_printoptions(threshold=np.inf)

from data_preprocessing import train_data
from functions import f, BCE, accuracy

BATCH = 16
EPOCH = 180
LR = 2e-2
EPS = 1e-7
ALPHA = 0


if __name__ == "__main__":
# Data Preprocessing
    _, train_x, train_y, val_data = train_data(sys.argv[1], sys.argv[2], sys.argv[3], val_split=0)
 
    train_x = np.concatenate((np.ones((train_x.shape[0], 1)).astype(float), train_x), axis=1)
    train_y = train_y.reshape(train_y.shape[0], 1)
    if val_data != None:
        val_data[0] = np.concatenate((np.ones((val_data[0].shape[0], 1)).astype(float), val_data[0]), axis=1)
        val_data[1] = val_data[1].reshape(val_data[1].shape[0], 1)

# Training
    w = np.ones((train_x.shape[-1], 1))
    dw2 = np.zeros((train_x.shape[-1], 1)) + EPS
    best_w = (w, -1, -1, -1)

    for i in range(EPOCH):
        index = np.random.permutation(range(train_x.shape[0]))
        train_x = train_x[index]
        train_y = train_y[index]
        for j in range(int(train_x.shape[0] / BATCH)):
            x_batch = train_x[j * BATCH: (j + 1) * BATCH]
            y_batch = train_y[j * BATCH: (j + 1) * BATCH]

            diff = y_batch - f(w, x_batch)
            dw = -np.dot(x_batch.T, diff) + ALPHA * np.vstack((np.array([[0.]]), w[1:]))
            if sys.argv[3]:
                dw2 += dw ** 2  # accumulate dw^2 for estimation of second derivative
            else:
                dw2 += 1
            w = w - LR / np.sqrt(dw2) * (dw)
        
        pred_y = f(w, train_x)
        loss = np.around(BCE(pred_y, train_y) / train_x.shape[0], decimals=4)
        acc = np.around(accuracy(np.around(pred_y), train_y), decimals=4)
        if val_data != None:
            pred_y = f(w, val_data[0])
            val_loss = np.around(BCE(pred_y, val_data[1]) / val_data[0].shape[0], decimals=4)
            val_acc = np.around(accuracy(np.where(pred_y >= 0.5, 1, 0), val_data[1]), decimals=4)
            print("iteration = {}, loss = {}, acc = {}, val_loss = {}, val_acc = {}".format(i, loss, acc, val_loss, val_acc))
            if val_acc >= best_w[-1]:
                best_w = (w, i, acc, val_acc)
        else:
            print("iteration = {}, loss = {}, acc = {}".format(i, loss, acc))
    if val_data != None:
        print(best_w[1:])
        np.save("model.logi", best_w[0])
    else:
        np.save("model.logi", w)
