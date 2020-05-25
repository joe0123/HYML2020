import numpy as np
import pandas as pd
EPS = 1e-10
DAY = 9 

DIM = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

def data_refreshing(data):
    data[1:] = np.where(data[1:] == 0, EPS, data[1:])
    for i in range(data.shape[0]):
        left = -1
        for j in range(data.shape[1]):
            if j > 0 and data[i][j] > 0 and data[i][j - 1] <= 0:
                if left == -1:
                    np.put(data[i], range(j), data[i][j])
                elif(left // 480 == j // 480):
                    for k in range(left + 1, j):
                        data[i][k] = (data[i][left] * (j - k) + data[i][j] * (k - left)) / (j - left)
                        #print(data[i][left], data[i][k], data[i][j])
                    left = -1

            if j < data.shape[1] - 1 and data[i][j] > 0 and data[i][j + 1] <= 0 and left == -1:
                left = j

        if left != -1:
            np.put(data[i], range(left + 1, data.shape[1]), data[i][left])

    return data

def x_standardizing(case, x, col):
    if case == "train":
        x_stat = np.array([np.mean(x, axis=0), np.std(x, axis=0)])
        np.save("stat", x_stat)
    else:
        try:
            x_stat = np.load("stat.npy")
        except:
            print("Go back to training!")
            exit()
    
    for i in col:
        if x_stat[1][i] != 0:
            x[:, i] = (x[:, i] - x_stat[0][i]) / x_stat[1][i]
    
    return x



def train_data(fn):
    df = pd.read_csv(fn, encoding="big5").iloc[:, 3:]
    df[df == 'NR'] = '0'
    data = [[] for i in range(18)]

    for i in range(len(df)):
        data[i % 18] += df.iloc[i].values.tolist()
    
    data = np.array(data).astype(float)
    data = data_refreshing(data)

    x = []
    y = []
    for h in range(data[0].shape[-1]):
        if h % 480 < DAY or len(np.argwhere(data[DIM, h-DAY: h + 1] <= 0)) > 0:
            continue
        x.append(data[DIM, h-DAY: h].T.flatten())
        y.append(data[9][h])
    
    x = np.array(x)
    y = np.array(y)
    x = x_standardizing("train", x, range(x.shape[-1]))
    
    #for i in range(x.shape[1]):
    #    print(i % 18, np.corrcoef(x[:, i].flatten(), y.flatten())[0][1])
    
    return x, y.reshape((y.shape[0], 1))



def test_data(fn):
    df = pd.read_csv(fn, header=None, encoding="big5").iloc[:, 2:]
    df[df == 'NR'] = '0'
    data = df.to_numpy().astype(float)
    test_x = []
    for i in range(int(data.shape[0] / 18)):
        test_x.append(data_refreshing(data[i * 18 + DIM]).T.flatten())
    
    test_x = np.array(test_x)

    test_x = x_standardizing("test", test_x, range(test_x.shape[-1]))
    
    return test_x

