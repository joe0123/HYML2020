import sys
import os
import torch
import torch.utils.data as data
import random

from config import configurations
from utils import *
from dataset import EN2CNDataset
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    print("\nLoading configurations...", flush=True)
    config = configurations(sys.argv)
    config.load_model = True
    print ('config:\n', vars(config))

    print("\nLoading data...", flush=True)
    test_dataset = EN2CNDataset(config.data_path, config.seq_len, "testing")
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    
    print("\nBuilding model...", flush=True)
    model, _ = build_model(config, test_dataset.en_size, test_dataset.cn_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\nStart testing...", flush=True)
    test_loss, bleu_score, result = test_model(model, test_loader, criterion, config.beam_width)

    print ("\nTest | loss = {:.4f}, Perplexity = {:.4f}, Bleu = {:.4f}".format(test_loss, np.exp(test_loss), bleu_score), flush=True)
    
    with open(sys.argv[3], 'w') as f:
        for i in result:
            print(' '.join(i[1]), file=f)
