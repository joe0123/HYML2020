import numpy as np
import random
import torch

seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from model import *


def build_model(config, en_size, cn_size):
    encoder = Encoder(en_size, config.emb_dim, config.hidden_size, config.n_layers, config.dropout)
    decoder = Decoder(cn_size, config.emb_dim, config.batch_size, config.seq_len, config.hidden_size * 2, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.cuda()

    return model, optimizer


def train_model(model, optimizer, train_iter, criterion, total_steps, recent_step, summary_steps, ss_case):
    model.train()
    model.zero_grad()
    
    loss_sum = 0.0
    train_loss = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.cuda(), targets.cuda()
        
        outputs, preds = model(sources, targets, schedule_sampling(ss_case, recent_step + step))
        
        optimizer.zero_grad()
        outputs = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum /= 5
            train_loss = loss_sum
            print("Step {}: Train | loss = {:.4f}, Perplexity = {:.4f}".format(recent_step + step + 1, loss_sum, np.exp(loss_sum)), flush=True)
            loss_sum = 0.0

    return model, optimizer, train_loss


def test_model(model, dataloader, criterion, k):
    model.eval()
    loss_sum, bleu_sum = 0.0, 0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        sources, targets = sources.cuda(), targets.cuda()
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets, k)
        
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = criterion(outputs, targets)
        loss_sum += loss.item()
        
        targets = targets.reshape(sources.size(0), -1)
        preds = tokens_sentences(preds, dataloader.dataset.int_cn)
        sources = tokens_sentences(sources, dataloader.dataset.int_en)
        targets = tokens_sentences(targets, dataloader.dataset.int_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        bleu_sum += computebleu(preds, targets)
        n += batch_size

    return loss_sum / len(dataloader), bleu_sum / n, result


def tokens_sentences(outputs, int_word):
    sts = []
    for tokens in outputs:
        st = []
        for token in tokens:
            w = int_word[str(int(token))]
            st.append(w)
            if w == '<EOS>':
                break
        sts.append(st)

    return sts


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

def schedule_sampling(case, recent_step):
    lower_bound = 0.5
    if case == "notf":
        return 0
    elif case == "tf":
        return 1
    elif case == "lin":
        return (max(0, 1 - (recent_step / 10000))) * (1 - lower_bound) + lower_bound
    elif case == "exp":
        return (0.75 ** (recent_step / 1000)) * (1 - lower_bound) + lower_bound
    elif case == "sig":
        k = 1114
        return (k / (k + np.exp(recent_step / k))) * (1 - lower_bound) + lower_bound
    else:
        print("\nSchedule sampling config error!", flush=True)
        exit()



import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score

def save_model(model, optimizer, store_model_path, recent_epoch):
    torch.save(model.state_dict(), "{}/model_{}.ckpt".format(store_model_path, recent_epoch))
    return

def load_model(model, load_model_path):
    print("Loading model from {}".format(load_model_path))
    model.load_state_dict(torch.load(load_model_path))
    return model


