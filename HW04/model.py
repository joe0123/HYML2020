seed = 1114
import random
random.seed(seed)
import torch
from torch import nn
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import warnings
warnings.filterwarnings('ignore')

class LSTM_Net(nn.Module):
    def __init__(self, embedding):
        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False # Frozen embedding layer when training

        self.embedding_dim = embedding.size(1)
        self.lstm = nn.GRU(input_size=self.embedding_dim, hidden_size=160, num_layers=2, dropout=0.5,
                batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(in_features=160*2, out_features=512),
                                        nn.PReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=512, out_features=1),
                                        nn.Sigmoid())
    def forward(self, inputs):
        feature, _ = self.lstm(self.embedding(inputs).float(), None)
        # hidden state = each cell's output
        # input = input, (initial hidden state, initial cell state)
        # input dim = (batch_size, seq_len, input_size). Note that we set batch first.
        # output = last layer's hidden state, (each layer's hidden state, each layer's cell state)
        # output dim = (batch_size, seq_len, num_directions, hidden_size). Note that we set batch first.
        return self.classifier(feature[:, -1, :].view(feature.size()[0], -1))

class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()

        self.classifier = nn.Sequential(nn.Linear(in_features=input_size, out_features=1024),
                                        nn.PReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=1024, out_features=2048),
                                        nn.PReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=2048, out_features=1),
                                        nn.Sigmoid())
    def forward(self, inputs):
        return self.classifier(inputs.float())
