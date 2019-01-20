"""
This is the python implementation of simple char rnn.
"""

import numpy as np

with open("data.txt", 'r') as f:
    data = f.read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

hidden_size = 100                                       # hidden dimensions
seq_length = 16                                         # steps to unroll for RNN
learning_rate = 1e-1                                    # learning rate

MAX_DATA = 1000000

Wxh = np.random.rand(hidden_size, vocab_size) * 0.01    # input to hidden
Whh = np.random.rand(hidden_size, hidden_size) * 0.01   # hidden to hidden
Why = np.random.rand(vocab_size, hidden_size) * 0.01    # hidden to output
bh = np.zeros((hidden_size, 1))                         # hidden bias
by = np.zeros((vocab_size, 1))                          # output bias


def lossFun(inputs, targets, hprev):
    """ Runs forward and backward pass through the RNN.

    Args:
    ----
        inputs: list of integers.For some i, inputs[i] is input character.
        targets: list of integers. targets[i] is the corresponding next character.
        hprev: initial hidden state array of shape [hidden_size, 1].
    Returns:
    -------
        loss: loss on the train data
        dWxh: gradient of Wxh w.r.t loss
        dWhh: gradient of Whh w.r.t loss
        dWhy: gradient of Why w.r.t loss
        dbh:  gradient of bh w.r.t loss
        dby:  gradient of by w.r.t loss
        h:    final hidden state
    """
    # cache the values during the forward pass at each time step
    # so that it can be reused during backward pass.
    xs, hs, ys, ps = {}, {}, {}, {}

    # initial hidden state
    hs[-1] = np.copy(hprev)

    # loss
    loss = 0

    # forward pass
    for t in range(len(inputs)):
        # input at time step t is xs[t]
        # prepare a one-hot encoded vector of shape (vocab_size, 1)
        # where inputs[t] is the index where the 1 goes in xs[t]
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        # compute h[t] from h[t-1] and x[t]
        hs[t] = np.tanh(np.dot(Whh, hs[t - 1]) + np.dot(Wxh, xs[t]) + bh)

        # compute y[y] from h[t]
        ys[t] = np.dot(Why, hs[t]) + by

        # compute p[t], softmax probabilites for output
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        # compute loss using cross entropy
        # targets is one-hot encoded, targets[t] has only one index
        # for ps[t] loss will be log of ps[t][index_where_targets[t]_is_non_zero]
        loss += -np.log(ps[t][targets[t], 0])

    # backward pass

    # gradients are initialized to zeros
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    # initialize the incoming gradient of h to zero
    dhnext = np.zeros_like(hs[0])

    # backward pass iterates over the input sequence backwards
    for t in reversed(range(len(inputs))):
        # y_hat - y
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        # gradient for Why => (y_hat - y) * h[t]
        dWhy += np.dot(dy, hs[t].T)
        # gradient for by => (y_hat - y)
        dby += dy

        # h[t] is sent to feed forward for calculation of y[t]
        # and also a state to next time step.
        # so gradient is sum of incoming gradient from fully-connected layer
        # and the incoming gradient from the next time step.
        # gradient of h => (Why * (y_hat - y)) + dh[t+1]
        dh = np.dot(Why.T, dy) + dhnext

        # for calculating the gradient of Whh, Wxh
        # backprop through tanh is required.
        # gradient => [(y_hat - y) * (Why) + dh[t+1] ]* (1 - h^2)
        dhraw = (1 - hs[t] * hs[t]) * dh

        # gradient for bh => dhraw
        dbh += dhraw
        # gradient for Whh[t] => dhraw * h[t-1]
        dWhh += np.dot(dhraw, hs[t - 1].T)
        # gradient for Wxh[t] => dhraw * x[t]
        dWxh += np.dot(dhraw, xs[t].T)

        # update the gradient of h to propagate to previous time step.
        dhnext = np.dot(Whh.T, dhraw)

    # gradient clipping to range [-5, 5]
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """ Run the RNN model in forward mode for n steps.

    Args:
    -----
        h: initial hidden state.
        seed_ix: seed letter for the first time step.
        n: number of steps to run the forward pass.

    Returns:
    --------
        Sequence of integers produced by the model.
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    # run the forward pass n times
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))

        # sample from the distribution produced by softmax
        # alternative can be greedy, take the ix which is max
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # prepare the input for next cell
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# n is the iteration counter, p is the input sequence pointer
n, p = 0, 0

# memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length

while p < MAX_DATA:
    if (p + seq_length + 1) >= len(data) or n == 0:
        # reset RNN memory
        hprev = np.zeros((hidden_size, 1))
        # go from start of the data
        p = 0

    # In each step we unroll the RNN for seq_length cells
    # and provide seq_length inputs and targets to learn.
    inputs = [char_to_ix[ch] for ch in data[p: p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1: p + seq_length + 1]]

    # sample from the model now and then.
    if n % 1000 == 0:
        # sample and print the output by the model.
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----------')
        print(txt)
        print('----------')

    # forward pass for sequence length of characters
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

    # update the weight parameters using adagrad optimization
    for param, dparam, mem in zip(
            [Wxh, Whh, Why, bh, by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param -= learning_rate * dparam / (np.sqrt(mem) + 1e-8)

    p += seq_length
    n += 1
