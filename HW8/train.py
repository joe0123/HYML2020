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
    print ('config:\n', vars(config))

    print("\nLoading data...", flush=True)
    train_dataset = EN2CNDataset(config.data_path, config.seq_len, "training")
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    val_dataset = EN2CNDataset(config.data_path, config.seq_len, "validation")
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    
    print("\nBuilding model...", flush=True)
    model, optimizer = build_model(config, train_dataset.en_size, train_dataset.cn_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters with {} trainable".format(total_param, trainable_param), flush=True)
    
    print("\nStart training...", flush=True)
    recent_step = 0
    train_losses = []
    val_losses = []
    bleu_scores = []
    while (recent_step < config.total_steps):
        model, optimizer, train_loss = train_model(model, optimizer, train_iter, criterion, config.total_steps, recent_step, config.summary_steps, config.ss_case)
        train_losses.append(train_loss)
        val_loss, bleu_score, result = test_model(model, val_loader, criterion, config.beam_width)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        recent_step += config.summary_steps
        print ("\nStep {}/{}: Val | loss = {:.4f}, Perplexity = {:.4f}, Bleu = {:.4f}".format(recent_step, config.total_steps, val_loss, np.exp(val_loss), bleu_score), flush=True)

        print(schedule_sampling(config.ss_case, recent_step))
        
        print("Saving model to {}/model_{}.ckpt...\n".format(config.store_model_path, recent_step))
        save_model(model, optimizer, config.store_model_path, recent_step)
        with open("{}/output_{}.txt".format(config.store_model_path, recent_step), 'w') as f:
            for i in result:
                for line in i:
                    print(line, file=f)
                print('\n', file=f)

    with open("{}/history.csv".format(config.store_model_path), 'w') as f:
        f.write("train,val,bleu\n")
        for i in range(len(train_losses)):
            f.write("{:.4f},{:.4f},{:.4f}\n".format(train_losses[i], val_losses[i], bleu_scores[i]))
