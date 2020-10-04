import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import heapq
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class Encoder(nn.Module):
    def __init__(self, en_size, emb_dim, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(en_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len]
        embedded = self.embedding(inputs)
        outputs, hidden = self.rnn(self.dropout(embedded))
        # GRU's input = input, initial hidden state
        # inputs_dim = [batch_size, seq_len, embed_dim]
        # GRU's output = last layer's hidden state over the sequence, each cell's hidden state after the last word
        # outputs_dim = [batch_size, seq_len, hidden_size * directions]
        # hidden_dim =  [n_layers * directions, batch_size, hidden_size]

        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, cn_size, emb_dim, batch_size, encoder_seq_len, hidden_size, n_layers, dropout, isatt):
        super().__init__()
        self.cn_size = cn_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_size, emb_dim)
        self.isatt = isatt
        if self.isatt:
            self.attention = Attention(self.hidden_size, batch_size, encoder_seq_len, self.n_layers)
            self.feature_dim = self.hidden_size * 2
        else:
            self.feature_dim = self.hidden_size
 
        self.rnn = nn.GRU(emb_dim, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(in_features=self.feature_dim, out_features=self.hidden_size*2),
                                        nn.Dropout(dropout),
                                        nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size * 4),
                                        nn.Dropout(dropout),
                                        nn.Linear(in_features=self.hidden_size * 4, out_features=self.cn_size))
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size, 1]
        # hidden: [batch_size, n_layers * directions, hidden_size]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input)) # [batch size, 1, emb_dim]
        output, hidden = self.rnn(self.prelu(embedded), hidden)
        # GRU's input = input, initial hidden state
        # input_dim = [batch_size, 1, emb_dim]
        # hidden_dim = [num_layers, directions, batch_size, hidden_size]
        # output = last layer's hidden state over the sequence, each cell's hidden state after the last word
        # outputs_dim = [batch_size, 1, hidden_size]
        # hidden_dim = [num_layers, batch_size, hidden_size]

        if self.isatt:
            atten = self.attention(encoder_outputs, output)
            output = torch.cat((output, atten), dim=-1)
        output = self.classifier(output.squeeze(1)) # [batch_size, vocab_size]
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size, encoder_seq_len, n_layers):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        #self.atten = nn.Linear(in_features=(encoder_seq_len * hidden_size + hidden_size), out_features=encoder_seq_len)
        self.W1 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.W2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.V = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
  
    def forward(self, encoder_outputs, decoder_output):
        # encoder_outputs: [batch_size, seq_len, encoder's hidden_size * directions]
        # decoder_output: [batch_size, 1, decoder's hidden_size]
        atten1 = encoder_outputs
        atten2 = decoder_output
        atten_weights = F.softmax(self.V(torch.tanh(self.W1(atten1) + self.W2(atten2))), dim=1)
        context_vector = torch.bmm(atten_weights.permute(0, 2, 1), encoder_outputs)
        return context_vector


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.n_layers == decoder.n_layers #Encoder and decoder must have equal number of layers!
            
    def forward(self, inputs, targets, tf_ratio):
        # inputs: [batch_size, seq_len]
        # targets: [batch_size, seq_len]
        batch_size = targets.size(0)
        target_len = targets.size(1)
        
    # Forward in encoder
        encoder_outputs, hidden = self.encoder(inputs)
    # Prepare decoder's initial hidden state
        hidden = hidden.reshape(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=-1)
    # Forward in decoder
        outputs = torch.zeros(batch_size, target_len, self.decoder.cn_size).cuda()
        preds = []
        for t in range(target_len - 1):
        # Find input vocab (teacher forcing or not)
            teacher_force = random.random() <= tf_ratio
            if t > 0 and not teacher_force:
                input = top1
            else:
                input = targets[:, t]
            output, hidden = self.decoder(input, hidden, encoder_outputs)
        # Record the original output over vocabs
            outputs[:, t + 1] = output
        # Record the predicted vocab (top one vocab)
            top1 = output.argmax(1)
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, inputs, targets, k):
        # inputs: [batch_size, seq_len]
        # targets: [batch_size, seq_len]
        batch_size = targets.size(0)
        target_len = targets.size(1)

    # Forward in encoder
        encoder_outputs, hidden = self.encoder(inputs)
    # Prepare decoder's initial hidden state
        hidden = hidden.reshape(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=-1)
    # Forward in decoder
        preds = [(0, [targets[:, 0]], [torch.zeros(batch_size, 1, self.decoder.cn_size).cuda()], hidden, False)]
        for t in range(target_len - 1):
            h = []
            for prev in preds:
                input = prev[1][-1]
                output, hidden = self.decoder(input, prev[3], encoder_outputs)
                pred = torch.topk(F.log_softmax(output, dim=1), k)
                values = pred.values.permute(1, 0)
                indices = pred.indices.permute(1, 0)
                for i in range(k):
                    #score = prev[0] + prev[0] / len(prev[1]) if prev[4] else prev[0] + values[i].item()
                    score = prev[0] + values[i].item()
                    if len(h) < k or (len(h) >= k and score > h[0][0]):
                        if len(h) >= k:
                            heapq.heappop(h)
                        heapq.heappush(h, (score, prev[1] + [indices[i]], prev[2] + [output.detach().unsqueeze(1)], hidden, indices[i] == 2))
                    else:
                        break
            assert len(h) == k
            preds = h.copy()
        preds = sorted(preds, reverse=True)[0]
        outputs = torch.cat(preds[2], dim=1)
        preds = torch.cat(preds[1][1:], dim=0).unsqueeze(0)
        
        return outputs, preds


