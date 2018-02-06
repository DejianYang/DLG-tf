# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from models.layers.rnn_encoder import RNNEncoder

x = tf.placeholder(tf.int32, [None, None])
length = tf.placeholder(tf.int32, [None])
embs = tf.get_variable("embeddings", [2000, 5])

emb_inp = tf.nn.embedding_lookup(embs, x)

encoder = RNNEncoder('gru', 'uni', 10)
outputs, state = encoder(emb_inp, length)
outputs2, state2 = encoder(emb_inp, length)

print(outputs.shape, state.shape)

bi_encoder = RNNEncoder('gru', 'bi', 10)
bi_outputs, bi_state = bi_encoder(emb_inp, length)

print(bi_outputs.shape, bi_state.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    np_x = np.asarray([[3, 4, 5, 0, 0], [4, 5, 1, 2, 0]])
    np_len = np.asarray([3, 4], dtype='int32')
    res = sess.run([outputs, state, outputs2, state2, bi_outputs, bi_state], {x: np_x, length: np_len})
    for r in res:
        print(r.shape)
        print(r)
