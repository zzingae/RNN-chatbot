import torch
import torch.nn as nn
import numpy as np

input_size=100
hidden_size=100
num_layers=2
x_len=5
y_len=6
batch=1

embeddings_matrix = np.load('./word2vec/naver_movie_embed.npy')
embeddings_matrix = torch.tensor(embeddings_matrix)
vocab_size = embeddings_matrix.shape[0]

# difference between Embedding and Linear: 
# Linear expects vectors (e.g. one-hot representation of the words), 
# Embedding expects tokens (e.g. words index)
word_embeddings = nn.Embedding(vocab_size, input_size).from_pretrained(embeddings_matrix)
word_embeddings.weight.requires_grad = False

rnn = nn.RNN(input_size, hidden_size, num_layers)

proj = nn.Linear(hidden_size, vocab_size)
proj.weight.data.copy_(embeddings_matrix)
proj.weight.requires_grad=False
sm = nn.Softmax(dim=2)

xs = torch.randint(low=0,high=10,size=(x_len,batch))
ys = torch.randint(low=0,high=10,size=(y_len,batch))
h0 = torch.randn(num_layers, batch, hidden_size)

# encoder
exs = word_embeddings(xs)
output, hn = rnn(exs, h0)
# decoder
eys = word_embeddings(ys)
output, hn = rnn(eys, hn)
logits = proj(output)
result = sm(logits)

print(logits.shape)
print(result.shape)