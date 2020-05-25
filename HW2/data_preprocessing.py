import numpy as np
np.random.seed(1114)
np.set_printoptions(threshold=np.inf)
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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

def x_normalizing(case, x, col):
    if case == "train":
        x_stat = np.array([np.min(x, axis=0), np.max(x, axis=0)])
        np.save("stat", x_stat)
    else:
        try:
            x_stat = np.load("stat.npy")
        except:
            print("Go back to training!")
            exit()
    
    for i in col:
        if (x_stat[1][i] - x_stat[0][i]) != 0:
            x[:, i] = (x[:, i] - x_stat[0][i]) / (x_stat[1][i] - x_stat[0][i])
    
    return x


#def feature_engineering(df, y):
def feature_engineering(df):
    df = df.rename(columns=lambda x: x.strip().lower())
    #df["result"] = y

    df.drop(["other rel <18 ever marr not in subfamily", "grandchild <18 never marr rp of subfamily"], inplace=True, axis=1)
    
# Age
    bins = [-1, 18, 25, 35, 45, 55, 65, 75, np.inf]
    col = "age"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="age", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)
        
# Capital gains
    bins = [-np.inf, 4600, 7600, 15000, np.inf] 
    col = "capital gains"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="capital_gain", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)

# Capital losses 
    bins = [-np.inf, 1400, 2000, 2200, 3200, np.inf] 
    col = "capital losses"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="capital_loss", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)

# Dividends
    bins = [-np.inf, 0, 5000, np.inf] 
    col = "dividends from stocks"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="dividend", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)

# Num persons worked for employer
    col = "num persons worked for employer"
    df = pd.concat([df, pd.get_dummies(df[col], prefix="worked_for_employer", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)

# Working Weeks
    bins = [-np.inf, 25, 45, np.inf] 
    col = "weeks worked in year"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="week", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)

# Wage per hour
    bins = [-np.inf, 0, 1200, 1800, 2200, np.inf] 
    col = "wage per hour"
    df[col] = pd.cut(df[col], bins)
    df = pd.concat([df, pd.get_dummies(df[col], prefix="wage", prefix_sep='_')], axis=1)
    df.drop(col, inplace=True, axis=1)
    
    #bins = [-np.inf] + [i for i in range(0, 15000, 200)] + [np.inf]
    #for i in range(len(bins) - 1):
    #    tmp0 = df[(df[col] <= bins[i]) | (df[col] > bins[i + 1])]["result"]
    #    tmp1 = df[(df[col] > bins[i]) & (df[col] <= bins[i + 1])]["result"]
    #    if len(tmp1) > 0:
    #        print(bins[i], bins[i + 1], len(tmp0), np.sum(tmp0) / len(tmp0), len(tmp1), np.sum(tmp1) / len(tmp1))
    #print(df.corr()["result"].sort_values(ascending=False))


    return df

noise = [0, 9, 14, 15, 17, 19, 24, 38, 42, 46, 47, 54, 57, 75, 76, 77, 80, 90, 98, 120, 139, 140, 144, 148, 150, 152, 154, 155, 157, 165, 170,174, 175, 178, 180, 181, 182, 193, 196, 197, 200, 211, 216, 218, 219, 225, 228, 229, 231, 235, 247, 250, 255, 259, 261, 267, 271, 273,274, 275, 279, 280, 282, 283, 284, 285, 286, 287, 293, 297, 298, 299, 300, 303, 306, 307, 309, 316, 320, 324, 326, 327, 329, 330, 331,332, 333, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 353, 354, 355, 356, 363, 364, 366, 370, 371, 372,374, 379, 385, 387, 389, 396, 402, 406, 408, 413, 423, 430, 431, 439, 440, 441, 447, 448, 451, 453, 464, 465, 468, 473, 474, 487, 488,489, 494, 499, 500, 514, 523, 524, 529, 531]

def train_data(x_fn, y_fn, best, val_split=0):
    df = pd.read_csv(x_fn).iloc[:, 1:]
    y = pd.read_csv(y_fn).iloc[:, 1:].to_numpy().astype(float)
    
    if best:
        #df = feature_engineering(df, y)
        df = feature_engineering(df)
        x = df.to_numpy().astype(float)
        x = np.delete(x, noise, axis=1)
    else:
        x = df.to_numpy().astype(float)
        #x = x_normalizing("train", x, range(x.shape[-1]))
        x = x_standardizing("train", x, range(x.shape[-1]))

    if val_split > 0:
        index = np.random.permutation(range(x.shape[0]))
        k = int(index.shape[0] * (1 - val_split))
        return df.columns.tolist(), x[index[:k]], y[index[:k]], [x[index[k:]], y[index[k:]]]
    else:
        return df.columns.tolist(), x, y, None


def test_data(fn, best):
    df = pd.read_csv(fn).iloc[:, 1:]
    if best:
        df = feature_engineering(df)
        x = df.to_numpy().astype(float)
        x = np.delete(x, noise, axis=1)
    else:
        x = df.to_numpy().astype(float)
        x = x_standardizing("train", x, range(x.shape[-1]))
    
    return df.columns.tolist(), x
