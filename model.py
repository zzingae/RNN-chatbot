import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_matrix=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if embed_matrix:
            self.embedding = nn.Embedding(vocab_size, self.hidden_size).from_pretrained(embed_matrix)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, self.hidden_size)

        self.rnn = nn.RNN(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        # try to put random vector to generate diverse output (zzingae)
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Chatbot(nn.Module):
    def __init__(self, hidden_size, embed_matrix=None):
        super(Chatbot, self).__init__()
        self.hidden_size = hidden_size

        if embed_matrix:
            self.vocab_size = embed_matrix.shape[0]
            self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size).from_pretrained(embed_matrix)
            self.word_embeddings.weight.requires_grad = False

        self.encoder = nn.RNN(self.hidden_size, self.hidden_size)
        self.decoder = nn.RNN(self.hidden_size, self.hidden_size)

        self.proj = nn.Linear(self.hidden_size, self.vocab_size)

        if embed_matrix:
            self.proj.weight.data.copy_(embed_matrix)
            self.proj.weight.requires_grad=False

    def forward(self, input):

        sm = nn.Softmax(dim=2)