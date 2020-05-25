import os
import numpy as np
import random
import re
import json
import torch
import torch.utils.data as data

seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class EN2CNDataset(data.Dataset):
    def __init__(self, root_dir, seq_len, set_name):
        self.dir = root_dir

        self.cn_int, self.int_cn = self.get_dictionary("cn")
        self.en_int, self.int_en = self.get_dictionary("en")

        self.data = []
        with open(os.path.join(self.dir, "{}.txt".format(set_name)), 'r') as f:
            for line in f:
                self.data.append(line)
        print("{} dataset size: {}".format(set_name, len(self.data)))
        
        self.cn_size = len(self.cn_int)
        self.en_size = len(self.en_int)
        self.seq_len = seq_len

    def get_dictionary(self, language):
        with open(os.path.join(self.dir, "word2int_{}.json".format(language)), 'r') as f:
            word_int = json.load(f)
        with open(os.path.join(self.dir, "int2word_{}.json".format(language)), 'r') as f:
            int_word = json.load(f)
        return word_int, int_word

    def transform(self, st, pad):
        return np.pad(st, (0, (self.seq_len - st.shape[0])), mode="constant", constant_values=pad)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sts = re.split("[\t\n]", self.data[index])
        sts = list(filter(None, sts))
        assert len(sts) == 2

        st = re.split(' ', sts[0])
        st = list(filter(None, st))
        en_st = [self.en_int['<BOS>']] + [self.en_int.get(w, self.en_int['<UNK>']) for w in st] + [self.en_int['<EOS>']]

        st = re.split(' ', sts[1])
        st = list(filter(None, st))
        cn_st = [self.cn_int['<BOS>']] + [self.cn_int.get(w, self.cn_int['<UNK>']) for w in st] + [self.cn_int['<EOS>']]

        en_st, cn_st = np.array(en_st), np.array(cn_st)
        en_st, cn_st = self.transform(en_st, self.en_int['<PAD>']), self.transform(cn_st, self.cn_int['<PAD>'])
        en_st, cn_st = torch.LongTensor(en_st), torch.LongTensor(cn_st)
        return en_st, cn_st
