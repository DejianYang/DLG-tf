# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.seq2seq as tc_seq2seq
from tensorflow.python.layers import core as layers_core
from models.model_base import *
from models.model_helper import *

class HREDModel(BaseTFModel):
    def __init__(self, config, mode, scope=None):
        super(HREDModel, self).__init__(config, mode, scope)

    def _build_graph(self):
        self._build_placeholders()
        self._build_embeddings()
        self._build_encoder()
        self._build_decoder()
        if self.mode != ModelMode.infer:
            self._compute_loss()
            if self.mode == ModelMode.train:
                self.create_optimizer(self.loss)
                # Summary
                self.train_summary = tf.summary.merge([
                                                          tf.summary.scalar("lr", self.learning_rate),
                                                          tf.summary.scalar("train_loss", self.loss),
                                                      ] + self.grad_norm_summary)
        pass

    def _build_placeholders(self):
        with tf.variable_scope("placeholders"):
            batch_size = None
            dialog_turn_size = None
            dialog_sent_size = None

            self.dialog_input = tf.placeholder(tf.int32,
                                                shape=[batch_size, dialog_turn_size, dialog_sent_size],
                                                name="dialog_inputs")
            self.dialog_length = tf.placeholder(tf.int32,
                                                      shape=[batch_size, dialog_turn_size],
                                                      name='dialog_input_lengths')

            self.dropout_keep_prob = tf.placeholder(tf.float32)

            self.batch_size = tf.shape(self.dialog_input)[0]
            self.turn_size = tf.shape(self.dialog_input)[1]
            self.sent_size = tf.shape(self.dialog_input)[2]
            self.dialog_turn_length = tf.reduce_sum(tf.sign(self.dialog_length), axis=1)
            pass

    def _build_embeddings(self):
        with tf.variable_scope("dialog_embeddings"):
            self.dialog_embeddings = tf.get_variable("dialog_embeddings",
                                                     shape=[self.config.vocab_size, self.config.emb_size],
                                                     dtype=tf.float32,
                                                     trainable=True)
            self._dialog_input_embs = tf.nn.embedding_lookup(self.dialog_embeddings,
                                                             tf.reshape(self.dialog_input,
                                                                        [self.batch_size*self.turn_size, self.sent_size]))
            print("dialog utterance embedding shape:", self._dialog_input_embs.shape)

    def _build_encoder(self):
        with tf.variable_scope("dialog_encoder"):
            with tf.variable_scope('utterance_rnn'):
                uttn_cell = get_rnn_cell(unit_type='gru',
                                         hidden_size=self.config.enc_hidden_size,
                                         num_layers=self.config.num_layers,
                                         dropout_keep_prob=self.dropout_keep_prob)
                _, uttn_states = tf.nn.dynamic_rnn(uttn_cell,
                                                   inputs=self._dialog_input_embs,
                                                   sequence_length=tf.reshape(self.dialog_length, [-1]),
                                                   dtype=tf.float32)

                uttn_states = tf.reshape(uttn_states, [self.batch_size,
                                                       self.turn_size,
                                                       self.config.enc_hidden_size])
                print('utterance shape', uttn_states.shape)

            with tf.variable_scope("context_rnn"):
                ctx_cell = get_rnn_cell(unit_type='gru',
                                        hidden_size=self.config.enc_hidden_size,
                                        num_layers=self.config.num_layers,
                                        dropout_keep_prob=self.dropout_keep_prob)

                ctx_outputs, ctx_state = tf.nn.dynamic_rnn(ctx_cell,
                                                           inputs=uttn_states,
                                                           sequence_length=self.dialog_turn_length,
                                                           dtype=tf.float32)
                self._enc_outputs = ctx_outputs
                self._enc_state = ctx_state

    @staticmethod
    def _compute_turn_loss(turn_logits, turn_output, length):
        batch_size = tf.shape(turn_output)[0]
        max_time = tf.shape(turn_output)[1]

        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=turn_output,
                                                                    logits=turn_logits)
        weights = tf.sequence_mask(length,
                                   maxlen=max_time,
                                   dtype=turn_logits.dtype)

        turn_loss = tf.reduce_sum(cross_loss * weights) / tf.to_float(batch_size)
        turn_ppl = tf.reduce_sum(cross_loss * weights)

        return turn_loss, turn_ppl

    def _build_decoder(self):
        with tf.variable_scope("dialog_decoder"):
            with tf.variable_scope("decoder_output_projection"):
                output_layer = layers_core.Dense(
                    self.config.vocab_size, use_bias=False, name="output_projection")

            with tf.variable_scope("decoder_rnn"):
                dec_cell = get_rnn_cell(unit_type='gru',
                                        hidden_size=self.config.dec_hidden_size,
                                        num_layers=self.config.num_layers,
                                        dropout_keep_prob=self.dropout_keep_prob)

                enc_outputs = self._enc_outputs #batch * turn * hidden_size
                enc_state = self._enc_state # batch * hidden_size

                # dialog input embeddings(batch_size * turn_size * sent_size* emb_size)
                dialog_emb_inp = tf.reshape(self._dialog_input_embs,
                                            [self.batch_size, self.turn_size,
                                             self.sent_size, self.config.emb_size])
                print("dialog emb shape", dialog_emb_inp.shape)

                # training
                if self.mode != ModelMode.infer: # not infer, do decode turn by turn
                    turn_index = tf.constant(0, dtype=tf.int32, name='turn_time_index')
                    turn_loss = tf.constant(0, dtype=tf.float32, name='turn_loss')

                    # turn total ppl
                    turn_ppl = tf.constant(0, dtype=tf.float32, name='turn_ppl')

                    def _decode_by_turn(idx, loss, ppl):
                        dec_init_state = enc_outputs[:, idx, :]
                        # batch* (sent_length -1) * emb_size
                        dec_emb_inp = dialog_emb_inp[:, idx+1, :-1, :]
                        print("dec emb input shape", dec_emb_inp.shape)
                        # remove eos to compute turn output length
                        turn_output_length = self.dialog_length[:, idx+1] - 1


                        dec_helper = tc_seq2seq.TrainingHelper(dec_emb_inp,
                                                               sequence_length=turn_output_length)

                        turn_decoder = tc_seq2seq.BasicDecoder(cell=dec_cell,
                                                               helper=dec_helper,
                                                               initial_state=dec_init_state,
                                                               output_layer=output_layer)

                        turn_dec_outputs, turn_dec_state, _ = tc_seq2seq.dynamic_decode(turn_decoder)
                        turn_logits = turn_dec_outputs.rnn_output
                        print('turn step output shape:', turn_logits.shape)

                        # Note that output max time is the max length
                        # remove sos as the actual output tokens
                        dec_output_tokens = self.dialog_input[:, idx + 1, 1:1+tf.shape(turn_logits)[1]]
                        # compute turn loss
                        _loss, _ppl = self._compute_turn_loss(turn_logits, dec_output_tokens, turn_output_length)
                        loss += _loss
                        ppl += _ppl
                        return [tf.add(idx, 1), loss, ppl]

                    dec_turn, total_loss, total_ppl = tf.while_loop(cond = lambda  i, l, p: tf.less(i, self.turn_size-1),
                                                             body = _decode_by_turn,
                                                             loop_vars=[turn_index, turn_loss, turn_ppl])
                    self._total_loss = total_loss
                    self._total_ppl = total_ppl
                    self.dec_turn = dec_turn
                else:
                    # do infer
                    beam_size = self.config.beam_size
                    length_penalty_weight = self.config.length_penalty_weight
                    maximum_iterations = tf.to_int32(self.config.infer_max_len)

                    start_tokens = tf.fill([self.batch_size], self.config.sos_idx)
                    end_token = self.config.eos_idx

                    dec_init_state = tc_seq2seq.tile_batch(enc_state, multiplier=beam_size)


                    # beam decoder
                    decoder = tc_seq2seq.BeamSearchDecoder(
                        cell=dec_cell,
                        embedding=self.dialog_embeddings,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=dec_init_state,
                        beam_width=beam_size,
                        output_layer=output_layer,
                        length_penalty_weight=length_penalty_weight)

                    dec_outputs, dec_state, _= tc_seq2seq.dynamic_decode(
                        decoder=decoder,
                        maximum_iterations=maximum_iterations)

                    self.predict_ids = dec_outputs.predicted_ids
        pass
    def _compute_loss(self):
        self.loss = self._total_loss
        self.ppl = self._total_ppl
        self.word_predict_count = tf.reduce_sum(self.dialog_length[:, 1:] - 1)
        pass

    def train(self, sess, batch_data):
        assert self.mode == ModelMode.train
        feed_dict = {
            self.dialog_input:batch_data.dialog,
            self.dialog_length:batch_data.dialog_length,
            self.dropout_keep_prob:self.config.dropout_keep_prob
        }

        res = sess.run([self.update_opt,
                        self.loss,
                        self.ppl,
                        self.word_predict_count,
                        self.batch_size,
                        self.train_summary,
                        self.global_step,
                        self.dec_turn
                        ], feed_dict)
        return res[1:]

    def eval(self, sess, batch_data):
        assert self.mode == ModelMode.eval
        feed_dict = {
            self.dialog_input: batch_data.dialog,
            self.dialog_length: batch_data.dialog_length,
            self.dropout_keep_prob: 1.0
        }
        res = sess.run([self.loss,
                        self.ppl,
                        self.word_predict_count,
                        self.batch_size,
                        self.global_step], feed_dict)
        return res

    def infer(self, sess, batch_data):
        assert self.mode == ModelMode.infer
        feed_dict = {
            self.dialog_input: batch_data.dialog,
            self.dialog_length: batch_data.dialog_length,
            self.dropout_keep_prob: 1.0
        }
        res = sess.run([self.predict_ids,
                       self.batch_size],
                       feed_dict)
        return res