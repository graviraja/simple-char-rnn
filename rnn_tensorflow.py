import tensorflow as tf
import numpy as np

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

tf.set_random_seed(42)

# declaration of inputs, targets and initial hidden state
inputs = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="inputs")
targets = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="targets")
initial_hidden_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")

# random initializer for the model weights with mean = 0.0 and std = 0.1
initializer = tf.random_normal_initializer(stddev=0.1)

with tf.variable_scope("RNN") as scope:
    hs_t = initial_hidden_state
    ys = []
    for t, xs_t in enumerate(tf.split(inputs, seq_length, axis=0)):
        if t > 0:
            scope.reuse_variables()
        Wxh = tf.get_variable("Wxh", [vocab_size, hidden_size], initializer=initializer)
        Whh = tf.get_variable("Whh", [hidden_size, hidden_size], initializer=initializer)
        Why = tf.get_variable("Why", [hidden_size, vocab_size], initializer=initializer)
        bh = tf.get_variable("bh", [hidden_size], initializer=initializer)
        by = tf.get_variable("by", [vocab_size], initializer=initializer)

        # forward pass

        # compute h[t] from h[t-1] and x[t]
        hs_t = tf.tanh(tf.matmul(xs_t, Wxh) + tf.matmul(hs_t, Whh) + bh)

        # compute y[t] from h[t]
        ys_t = tf.matmul(hs_t, Why) + by
        ys.append(ys_t)

# final hidden state after the forward pass
hprev = hs_t

# softmax op for sampling
# applying softmax for the last time step
output_softmax = tf.nn.softmax(ys[-1])
outputs = tf.concat(ys, axis=0)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs))
# optimization
minimizer = tf.train.AdamOptimizer()

# backward pass
grads_and_vars = minimizer.compute_gradients(loss)

# gradient clipping to range [-5, 5]
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
    clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
    clipped_grads_and_vars.append((clipped_grad, var))

# weight update
updates = minimizer.apply_gradients(clipped_grads_and_vars)


# convert the data to one-hot encoding
def one_hot(v):
    return np.eye(vocab_size)[v]

# start the tensorflow session
sess = tf.Session()
# initialize all the variables with their initializers
init = tf.global_variables_initializer()
sess.run(init)

# n is the iteration counter, p is the input sequence pointer
n, p = 0, 0

# initial hidden state
hprev_val = np.zeros([1, hidden_size])

while True:
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev_val = np.zeros([1, hidden_size])
        p = 0

    # reading sequence length inputs and targets
    input_vals = [char_to_ix[ch] for ch in data[p: p + seq_length]]
    output_vals = [char_to_ix[ch] for ch in data[p + 1: p + seq_length + 1]]

    # converting the data to one-hot representation
    input_data = one_hot(input_vals)
    output_data = one_hot(output_vals)

    # running the forward pass for the input and targets
    hprev_val, loss_val, _ = sess.run(
        [hprev, loss, updates],
        feed_dict={
            inputs: input_data,
            targets: output_data,
            initial_hidden_state: hprev_val
        })

    # sample from the model now and then
    if n % 500 == 0:
        sample_length = 200
        start_index = input_vals[0]
        sample_sequence = [char_to_ix[ch] for ch in data[start_index: start_index + seq_length]]
        ixes = []
        sample_prev_state = np.copy(hprev_val)

        for t in range(sample_length):
            sample_input_vals = one_hot(sample_sequence)
            sample_output_softmax_val, sample_prev_state_val = \
                sess.run([output_softmax, hprev],
                         feed_dict={inputs: sample_input_vals, initial_hidden_state: sample_prev_state})

            ix = np.random.choice(range(vocab_size), p=sample_output_softmax_val.ravel())
            ixes.append(ix)
            sample_sequence = sample_sequence[1:] + [ix]

        txt = ''.join(ix_to_char[ix] for ix in ixes)
        print('----\n %s \n----\n' % (txt,))

    p += seq_length
    n += 1
