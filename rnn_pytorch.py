"""
This is the pytorch implementation of simple char rnn.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable

with open('data.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

hidden_size = 100                                       # hidden dimensions
seq_length = 16                                         # steps to unroll for RNN
learning_rate = 1e-1                                    # learning rate

MAX_DATA = 1000000


# convert the data to one-hot encoding
def one_hot(v):
    return torch.eye(vocab_size)[v]


class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=100):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.Wxh = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.Whh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.Why = Parameter(torch.Tensor(self.hidden_size, self.vocab_size))
        self.bh = Parameter(torch.zeros(self.hidden_size))
        self.by = Parameter(torch.zeros(self.vocab_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.vocab_size)
        stdv2 = 1. / math.sqrt(self.hidden_size)
        self.Wxh.data.uniform_(-stdv, stdv)
        self.Whh.data.uniform_(-stdv2, stdv2)
        self.Why.data.uniform_(-stdv, stdv)
    
    def forward(self, hprev, x):
        # forward pass
        outputs = []
        for i in range(len(x)):
            hs_t = torch.tanh(torch.mm(x[i].view(1, -1), self.Wxh) + torch.mm(hprev, self.Whh) + self.bh)
            outputs.append(hs_t)
        output = torch.cat(outputs)
        logits = torch.mm(output, self.Why) + self.by
        return logits, output[-1].view(1, -1)

model = CharRNN(vocab_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length


n, p = 0, 0

while p < MAX_DATA:
    if (p + seq_length + 1) >= len(data) or n == 0:
        # reset RNN memory
        hprev = torch.zeros((1, hidden_size))
        # go from start of the data
        p = 0

    # In each step we unroll the RNN for seq_length cells
    # and provide seq_length inputs and targets to learn.
    inputs = [char_to_ix[ch] for ch in data[p: p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1: p + seq_length + 1]]
    input_vals = one_hot(inputs)
    target_vals = torch.tensor(targets)

    if n % 500 == 0:
        # sample and print the output by the model.
        sample_length = 200
        start_index = inputs[0]
        sample_sequence = [char_to_ix[ch] for ch in data[start_index: start_index + seq_length]]
        ixes = []
        sample_prev_state = torch.clone(hprev)
        
        for t in range(sample_length):
            with torch.no_grad():
                sample_input_vals = one_hot(sample_sequence)
                sample_output_softmax_val, sample_prev_state_val = model(sample_prev_state, sample_input_vals)
                last_prediction = nn.Softmax(dim=0)(sample_output_softmax_val[-1])
                ix = np.random.choice(range(vocab_size), p=last_prediction.data.numpy().ravel())
                ixes.append(ix)
                sample_sequence = sample_sequence[1:] + [ix]
                sample_prev_state = sample_prev_state_val

        txt = ''.join(ix_to_char[ix] for ix in ixes)
        print('----\n %s \n----\n' % (txt,))

    # forward pass for sequence length of characters
    model.zero_grad()
    logits, hprev = model(hprev, input_vals)
    # detach the hprev, otherwise loss.backward() is trying to back-propagate all the way through to the start of time, 
    # which works for the first batch but not for the second because the graph for the first batch has been discarded.
    hprev = hprev.detach()
    loss = criterion(logits, target_vals)
    # replace loss.backward() with loss.backward(retain_graph=True) but know that each successive batch will take more 
    # time than the previous one because it will have to back-propagate all the way through to the start of the first batch.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    p += seq_length
    n += 1
