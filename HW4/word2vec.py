import os
from gensim.models import word2vec
import numpy as np

from utils import train_data, test_data

DIM = 256

if __name__ == "__main__":
    print("Loading train data ...", flush=True)
    train_x1, _ = train_data("data/training_label.txt", True)
    train_x0 = train_data("data/training_nolabel.txt", False)
    
    print("Loading test data ...", flush=True)
    test_x = test_data("data/testing_data.txt")
    #print(np.percentile([len(i) for i in train_x1], 75))
    #print(np.percentile([len(i) for i in train_x0], 75))
    #print(np.percentile([len(i) for i in test_x], 75))
    #exit()

    print("Word2Vec ...", flush=True)
    model = word2vec.Word2Vec(train_x1 + train_x0 + test_x, size=256, window=5, min_count=5, workers=12, iter=10, sg=1)
    
    print("Saving model ...", flush=True)
    model.save("w2v_model/w2v.model")
