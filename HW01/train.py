import sys
import numpy as np
np.random.seed(1114)
np.set_printoptions(threshold=np.inf)

EPS = 1e-10
DAY = 9

from data_preprocessing import train_data


def Adagrad(x, y, val_split=0, epochs=10000, lr=10, alpha=10):
    if val_split > 0:
        index = np.random.permutation(range(x.shape[0]))
        k = int(index.shape[0] * (1 - val_split))
        train_x = x[index[:k]]
        train_y = y[index[:k]]
        val_x = x[index[k:]]
        val_y = y[index[k:]]
    else:
        train_x = x
        train_y = y
    
    w = np.ones((train_x.shape[-1], 1))
    dw2 = np.zeros((train_x.shape[-1], 1)) + EPS
    best_w = (w, -1, np.inf, np.inf)
    loss = []
    val_loss = []
    for i in range(epochs):
        diff = train_y - np.dot(train_x, w)
        dw = -np.dot(train_x.T, diff) + alpha * np.vstack((np.array([[0.]]), np.sign(w[1:])))
        dw2 += dw ** 2  # accumulate dw^2 for estimation of second derivative
        w -= lr * dw / np.sqrt(dw2)

        diff = train_y - np.dot(train_x, w)
        loss.append(np.sum(diff ** 2) / train_x.shape[0])
        if val_split > 0:
            val_diff = val_y - np.dot(val_x, w)
            val_loss.append(np.sum(val_diff ** 2) / val_x.shape[0])
            print("iteration = {}, loss = {}, val_loss = {}".format(i, loss[-1], val_loss[-1]))
            if val_loss[-1] < best_w[-1]:
                best_w = (w, i, loss[-1], val_loss[-1])
        else:
            print("iteration = {}, loss = {}".format(i, loss[-1]))
    if val_split > 0:
        print(best_w[1:])
        np.save("model.adagrad", best_w[0])
        return loss, val_loss
    else:
        np.save("model.adagrad", w)
        return loss

def svr(x, y):
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    import pickle

    model = SVR(kernel="linear", epsilon=5).fit(x, y)
    with open("model.svr.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(cross_val_score(model, x, y, cv=KFold(n_splits=3, shuffle=True)))
    #print(cross_val_score(model, x, y, cv=KFold(n_splits=3, shuffle=True), scoring="neg_mean_squared_error"))

    return


if __name__ == "__main__":
    x, y = train_data(sys.argv[1])
    #print(x.shape, y.shape)
    #print(np.mean(x), np.std(x))

# Training
    if sys.argv[2] == "adagrad":
        x = np.concatenate((np.ones((x.shape[0], 1)).astype(float), x), axis=1)
        loss = Adagrad(x, y)
    elif sys.argv[2] == "svr":
        svr(x, y)
