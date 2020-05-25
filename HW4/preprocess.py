seed = 1114
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import torch
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

class Preprocess():
    def __init__(self, st_len, w2v_path):
        self.w2v_path = w2v_path
        self.st_len = st_len
        self.index_word = []
        self.word_index = {}
        self.em_matrix = []
    def add_random_em(self, w):
        v = np.random.uniform(size=self.em_dim)
        self.word_index[w] = len(self.word_index)
        self.index_word.append(w)
        self.em_matrix.append(v)
    def make_em(self):
    # Load word2vec.model
        self.em = Word2Vec.load(self.w2v_path)
        self.em_dim = self.em.vector_size
    # Construct embedding matrix
        self.add_random_em("<PAD>")
        self.add_random_em("<UNK>")
        for i, w in enumerate(self.em.wv.vocab):
            self.word_index[w] = len(self.word_index)
            self.index_word.append(w)
            self.em_matrix.append(self.em[w])
        self.em_matrix = torch.tensor(self.em_matrix)
        self.em_matrix = (self.em_matrix - torch.mean(self.em_matrix, axis=0)) / torch.std(self.em_matrix, axis=0)
        return self.em_matrix
    def trun_pad_seq(self, st):
        if len(st) >= self.st_len:
            st = st[:self.st_len]
        else:
            pad_len = self.st_len - len(st)
            for _ in range(pad_len):
                st.append(self.word_index["<PAD>"])
        assert len(st) == self.st_len
        return st
    def st_index(self, sts):
        st_list = []
        known = 0
        unknown = 0
        for st in sts:
            tmp = []
            for w in st:
                if w in self.word_index:
                    known += 1
                    tmp.append(self.word_index[w])
                else:
                    unknown += 1
                    tmp.append(self.word_index["<UNK>"])
            tmp = self.trun_pad_seq(tmp)
            st_list.append(tmp)
        print("{:.02f}% of words are known".format(known / (known + unknown) * 100), flush=True)
        return torch.LongTensor(st_list)
    def labels(self, y):
        return torch.LongTensor(y)

