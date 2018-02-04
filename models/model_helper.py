# -*- coding:utf-8 -*-
import time
import tensorflow as tf
import tensorflow.contrib as tc


def get_rnn_cell(unit_type, hidden_size, num_layers=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        unit_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        num_layers: MultiRNNCell are used if num_layers > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    if unit_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif unit_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif unit_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(unit_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    if num_layers > 1:
        cell = tc.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    return cell


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=1)
    return kld


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("loaded %s model parameters from %s, time %.2fs" %
          (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        print("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step
