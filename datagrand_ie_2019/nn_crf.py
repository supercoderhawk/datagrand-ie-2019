# -*- coding: utf-8 -*-
"""Builds network of NNCRF, including training and inference,"""
import math
import time
import os
import numpy as np
import tensorflow as tf
from pysenal.utils import get_chunk
from .nn_crf_config import NeuralNetworkCRFConfig
from .utils.constant import (FIT, INFERENCE, LOSS_LOG_LIKELIHOOD, LOSS_MAX_MARGIN,
                             NNCRF_DROPOUT_EMBEDDING, NNCRF_DROPOUT_HIDDEN, MODEL_DIR)
from .nn_crf_base import NeuralNetworkCRFBase
from .elmo_loader import ELMO


class NeuralNetworkCRF(NeuralNetworkCRFBase):
    def __init__(self, mode, dest_dir, config: NeuralNetworkCRFConfig, label2result):
        super().__init__(mode=mode, config=config)
        self.dest_dir = dest_dir
        self.label2result = label2result
        self.params = {}
        self.build_network()

    def build_network(self):
        """
        build the total network framework of NN-CRF, including initialize placeholders and variables,
        create network architecture,
        :return:
        """
        with self.graph.as_default():
            self.init_placeholder()
            self.elmo = ELMO(self.sess, self.input_word_ids)
            self.embedding_layer = self.get_embedding_layer()
            if self.mode == FIT:  # and self.config.dropout_position == NNCRF_DROPOUT_EMBEDDING:
                self.embedding_layer = self.get_dropout_layer(self.embedding_layer)
            self.hidden_layer = self.get_neural_network_layer(self.embedding_layer)
            if self.mode == FIT:  # and self.config.dropout_position == NNCRF_DROPOUT_HIDDEN:
                self.hidden_layer = self.get_dropout_layer(self.hidden_layer)
            self.output = self.get_full_connected_layer(self.hidden_layer)
            # self.output = tf.nn.softmax(self.output)
            self.init_crf_variable()

            if self.mode == FIT:
                regularizer = tf.contrib.layers.l2_regularizer(self.config.regularization_rate)
                self.regularization = tf.contrib.layers.apply_regularization(regularizer,
                                                                             tf.trainable_variables())
                if self.config.loss_function_name == LOSS_LOG_LIKELIHOOD:
                    self.loss = self.get_log_likehood_loss()
                elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                    self.loss = self.get_max_margin_loss()
                else:
                    raise Exception('loss function is not supported.')
                self.loss += self.regularization
                # self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                # self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                # self.optimizer = tf.train.AdagradOptimizer(self.config.learning_rate)
                gvs = self.optimizer.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_value(grad, -30, 30), var) for grad, var in gvs]
                self.train_model = self.optimizer.apply_gradients(capped_gvs)
                # self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
                # self.optimizer = tf.train.AdagradOptimizer(self.config.learning_rate)
                # self.train_model = self.optimizer.minimize(self.loss)
                self.sess.run(tf.global_variables_initializer())
                # self.loss_summary = tf.summary.scalar('loss', self.loss)
            elif self.mode == INFERENCE:
                self.sess.run(tf.global_variables_initializer())
                filename = MODEL_DIR + 'nn/' + self.config.model_name + '.ckpt'
                # print(self.config.hidden_layers)
                # print(filename)
                tf.train.Saver().restore(self.sess, filename)

    def fit(self):
        print('start traininig......')
        with self.sess as sess:
            tf.global_variables_initializer().run()
            if self.config.loss_function_name == LOSS_LOG_LIKELIHOOD:
                self.fit_log_likehood(sess)
            elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                self.fit_max_margin(sess)
            else:
                raise Exception('loss function is not supported.')

    def inference(self, data):
        if type(data) not in {str, list}:
            raise Exception('Input data type error, not string or list')
        elif isinstance(data, str):
            data = [data]

        sent_lengths = [len(sent['tokens']) for sent in data]
        sent_tokens = [[t['text'] for t in sent['tokens']] for sent in data]
        input_indices = self.sentences2input_indices(sent_tokens, max(sent_lengths))
        runner = [self.output, self.transition]  # , self.init_transition]

        output = []
        for input_chunk,len_chunk in zip(get_chunk(input_indices,50),get_chunk(sent_lengths,50)):
            feed_dict = {self.input_word_ids: input_chunk,
                         self.seq_length: len_chunk,
                         self.inference_batch_size: len(len_chunk)}
            batch_output, transition = self.sess.run(runner, feed_dict=feed_dict)
            output.extend(batch_output)

        results = []
        for sent, sent_output in zip(data, output):
            labels = self.viterbi(sent_output.T[:, :len(sent['tokens'])], transition)
            entities = self.label2result(sent['tokens'], labels, self.label_schema)
            sent['entities'] = entities
            sent['labels'] = labels
            results.append(sent)
        return results

    def fit_log_likehood(self, sess, interval=1):
        saver = tf.train.Saver(max_to_keep=100)
        training_data = self.data.mini_batch(self.config.batch_size)
        for index, (words, labels, seq_lengths) in enumerate(training_data):
            # print(len(words[0]))
            if index and index % self.batch_count == 0:
                epoch = index // self.batch_count
                print('epoch {0}'.format(epoch))
                if epoch > 0 and epoch % interval == 0:
                    basename = self.dest_dir + self.config.model_name
                    saver.save(sess, basename + '.ckpt')
                    self.config.to_json(basename + '.json')
                if epoch > 100:
                    break
            if index < 1000:
                lr = self.config.learning_rate * 10
            elif index < 10000:
                lr = self.config.learning_rate * 5
                # elif index< 10000:
                # lr = self.config.learning_rate * 10
            else:
                lr = self.config.learning_rate
            feed_dict = {self.input_word_ids: words, self.true_labels: labels,
                         self.seq_length: seq_lengths, self.lr: lr}
            _, loss = self.sess.run([self.train_model, self.loss], feed_dict=feed_dict)
            if index % 50 == 0:
                print(loss)

    def get_log_likehood_loss(self):
        with tf.name_scope('log_likehood'):
            crf_loss, _ = tf.contrib.crf.crf_log_likelihood(self.output,
                                                            self.true_labels,
                                                            self.seq_length,
                                                            self.transition)
            return -tf.math.reduce_sum(crf_loss) / self.config.batch_size

    def fit_max_margin(self, sess, interval=1):
        saver = tf.train.Saver(max_to_keep=100)
        training_data = self.data.mini_batch(self.config.batch_size)
        start = time.time()
        for index, (words, labels, seq_lengths) in enumerate(training_data):
            if index and index % self.batch_count == 0:
                epoch = index // self.batch_count
                print('epoch {0}'.format(epoch))
                print((time.time() - start) / 60)
                start = time.time()
                if epoch > 0 and epoch % interval == 0:
                    basename = self.dest_dir + self.config.model_name
                    saver.save(sess, basename + '.ckpt')
                    self.config.to_json(basename + '.json')
                if epoch > 100:
                    break
            transition = self.transition.eval(session=sess)
            init_transition = self.init_transition.eval(session=sess)
            feed_dict = {self.input_word_ids: words, self.seq_length: seq_lengths}
            output = sess.run(self.output, feed_dict=feed_dict)
            pred_seq = []

            for i in range(self.config.batch_size):
                state = output[i, :seq_lengths[i], :].T
                seq = self._viterbi_training_stage(state, transition, init_transition, labels[i],
                                                   self.config.batch_length)
                pred_seq.append(seq)

            feed_dict = {self.true_seq: labels, self.pred_seq: pred_seq,
                         self.input_word_ids: words, self.seq_length: seq_lengths}
            sess.run(self.train_model, feed_dict=feed_dict)
            transition_diff = sess.run(self.transition_difference, feed_dict=feed_dict)
            state_diff = sess.run(self.state_difference, feed_dict=feed_dict)

            if index and index % 100 == 0:
                # print('======================')
                # print('batch index {0}'.format(index // self.batch_count))
                # print('transition: ', np.sum(transition_diff) / self.config.batch_size)
                # print('state difference: ', np.sum(state_diff) / self.config.batch_size)
                print('loss: ', sess.run(self.loss, feed_dict=feed_dict))

    def get_max_margin_loss(self):
        with tf.name_scope('max_margin'):
            batch_index = np.repeat(np.expand_dims(np.arange(0, self.config.batch_size), 1),
                                    self.config.batch_length, 1)
            sent_index = np.repeat(np.expand_dims(np.arange(0, self.config.batch_length), 0),
                                   self.config.batch_size, 0)
            self.true_index = tf.stack([batch_index, sent_index, self.true_seq], axis=2)
            self.pred_index = tf.stack([batch_index, sent_index, self.pred_seq], axis=2)
            pred_state = tf.gather_nd(self.output, self.pred_index)
            true_state = tf.gather_nd(self.output, self.true_index)

            self.state_difference = tf.reduce_sum(pred_state - true_state, axis=1)
            pred_transition = tf.gather_nd(self.transition,
                                           tf.stack([self.pred_seq[:, :-1], self.pred_seq[:, 1:]], 2))
            true_transition = tf.gather_nd(self.transition,
                                           tf.stack([self.true_seq[:, :-1], self.true_seq[:, 1:]], 2))
            transition_mask = tf.sequence_mask(self.seq_length - 1,
                                               self.config.batch_length - 1,
                                               dtype=tf.float32)
            pred_transition = transition_mask * pred_transition
            true_transition = transition_mask * true_transition
            self.transition_difference = tf.reduce_sum(pred_transition - true_transition, axis=1)
            pred_init_transition = tf.gather_nd(self.init_transition, tf.expand_dims(self.pred_seq[:, 0], 1))
            true_init_transition = tf.gather_nd(self.init_transition, tf.expand_dims(self.true_seq[:, 0], 1))
            self.init_transition_difference = pred_init_transition - true_init_transition
            score_diff = self.state_difference + self.transition_difference + self.init_transition_difference
            hinge_loss = self.config.hinge_rate * tf.count_nonzero(self.pred_seq - self.true_seq, axis=1,
                                                                   dtype=tf.float32)
            # loss = tf.reduce_sum(tf.maximum(score_diff, -hinge_loss)) / self.config.batch_size
            loss = tf.reduce_sum(score_diff) / self.config.batch_size
            return loss

    def get_cross_entropy_softmax_loss(self):
        with tf.name_scope('cross_entropy_softmax'):
            pass

    def init_placeholder(self):
        self.input_word_ids = tf.placeholder(tf.int32, [None, None])
        self.true_labels = tf.placeholder(tf.int32, [None, None])
        self.seq_length = tf.placeholder(tf.int32, [None])
        if self.mode == FIT:
            self.true_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.batch_length])
            self.pred_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.batch_length])
            output_shape = [self.config.batch_size, self.config.batch_length, self.label_count]
            self.output_placeholder = tf.placeholder(tf.float32, output_shape)
            self.lr = tf.placeholder(tf.float32, None)
        elif self.mode == INFERENCE:
            self.inference_batch_size = tf.placeholder(tf.int32, shape=())

    def init_crf_variable(self):
        with tf.variable_scope('crf'):
            self.transition = tf.Variable(
                tf.random_uniform([self.label_count, self.label_count], -0.001, 0.001),
                name='transition')
            if self.config.loss_function_name == 'log likehood':
                self.init_transition = tf.Variable(tf.zeros([self.label_count]), name='init_transition',
                                                   trainable=False)
            elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                self.init_transition = tf.Variable(tf.random_uniform([self.label_count], -0.001, 0.001),
                                                   name='init_transition')

    def get_embedding_layer(self):
        with tf.variable_scope('embedding'):
            embed_path = self.config.embed_path
            if embed_path is not None:# and os.path.exists(embed_path):
                # pretrained_embedding = self.vocab.load_glove_embedding(embed_path)
                pretrained_embedding = self.vocab.load_word2vec_embedding(embed_path)
                embeddings = tf.Variable(pretrained_embedding, name='embeddings', dtype=tf.float32)
            else:
                embeddings = self.__get_variable([self.dict_size, self.config.word_embed_size], 'embeddings')
                # embeddings = self.__get_variable([4552, self.config.word_embed_size], 'embeddings')
            self.params['embedding'] = embeddings
            # if self.mode == FIT:
            embedding_input = tf.nn.embedding_lookup(embeddings, self.input_word_ids)
            elmo_input = self.elmo.elmo_output['weighted_op']
            # print(embedding_input.shape)
            # print(elmo_input.shape)
            layer = tf.concat([embedding_input, elmo_input], axis=-1)
                # tf.nn.embedding_lookup(embeddings, self.input_word_ids)
                # input_size = [self.config.batch_size, self.config.batch_length,
                #               self.config.concat_embed_size]
                # layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input_word_ids), input_size)
            # else:
            #     layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input_word_ids),
            #                        [self.inference_batch_size, -1, self.config.concat_embed_size])
            return layer

    def get_neural_network_layer(self, embedding_layer):
        with tf.variable_scope('neural_network'):
            hidden_layers = self.config.hidden_layers
            hidden_layer = self.__get_layer_by_type(hidden_layers[0]['type'], embedding_layer,
                                                    hidden_units=hidden_layers[0]['units'])
            for layer in hidden_layers[1:]:
                hidden_layer = self.__get_layer_by_type(layer['type'], hidden_layer,
                                                        hidden_units=layer['units'])
            return hidden_layer

    def __get_layer_by_type(self, type_name, layer, **kwargs):
        if 'hidden_units' not in kwargs:
            raise Exception('don\'t assign hidden units')

        hidden_units = kwargs['hidden_units']

        if type_name == 'mlp':
            return self.get_mlp_layer(layer, hidden_units)
        elif type_name == 'rnn':
            return self.get_rnn_layer(layer, hidden_units)
        elif type_name == 'lstm':
            return self.get_lstm_layer(layer, hidden_units)
        elif type_name == 'gru':
            return self.get_gru_layer(layer, hidden_units)
        elif type_name == 'bidirectional_lstm':
            return self.get_bidirectional_lstm_layer(layer, hidden_units)
        elif type_name == 'bidirectional_gru':
            return self.get_bidirectional_gru_layer(layer, hidden_units)
        else:
            raise ValueError('error type name')

    def get_mlp_layer(self, layer, hidden_units, name='mlp',
                      weight_name='hidden_weight', bias_name='hidden_bias'):
        hidden_weight = self.__get_variable([hidden_units, self.config.concat_embed_size], name=weight_name)
        hidden_bias = tf.Variable(tf.random_uniform([hidden_units, 1, 1], -0.01, 0.01), name=bias_name)
        self.params['hidden_weight'] = hidden_weight
        self.params['hidden_bias'] = hidden_bias
        layer = tf.sigmoid(
            tf.tensordot(hidden_weight, tf.transpose(layer), [[1], [0]]) + hidden_bias,
            name=name)
        return tf.transpose(layer)

    def get_rnn_layer(self, layer, hidden_units, name='rnn'):
        rnn = tf.nn.rnn_cell.BasicRNNCell(hidden_units)
        rnn_output, rnn_out_state = tf.nn.dynamic_rnn(rnn, layer, dtype=tf.float32)
        return rnn_output

    def get_lstm_layer(self, layer, hidden_units, name='lstm'):
        lstm = tf.nn.rnn_cell.LSTMCell(hidden_units)
        lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, layer,
                                                        sequence_length=self.seq_length, dtype=tf.float32)
        return lstm_output

    def get_bidirectional_lstm_layer(self, layer, hidden_units, name='bidirectional_lstm'):
        lstm_fws = []
        lstm_bws = []
        for i in range(1):
            lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_units // 2)
            lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_units // 2)
            lstm_fws.append(lstm_fw)
            lstm_bws.append(lstm_bw)
        func = tf.contrib.rnn.stack_bidirectional_dynamic_rnn
        bilstm_output, output_state_fw, output_state_bw = func(lstm_fws, lstm_bws,
                                                               layer, sequence_length=self.seq_length,
                                                               dtype=tf.float32)
        # bilstm_output, bilstm_output_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw,
        #  layer, self.seq_length,
        #  dtype=tf.float32)
        # return tf.concat([bilstm_output[0], bilstm_output[1]], -1)
        # print(bilstm_output.shape)
        # print(bilstm_output[:,:,-1].shape)
        return bilstm_output

    def get_gru_layer(self, layer, hidden_units, name='gru'):
        gru = tf.nn.rnn_cell.GRUCell(hidden_units)
        gru_output, gru_out_state = tf.nn.dynamic_rnn(gru, layer, dtype=tf.float32)
        return gru_output

    def get_bidirectional_gru_layer(self, layer, hidden_units, name='bidirectional_gru'):
        gru_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_units // 2)
        gru_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_units // 2)
        gru_output, gru_output_state = tf.nn.bidirectional_dynamic_rnn(gru_fw, gru_bw, layer, self.seq_length,
                                                                       dtype=tf.float32)
        return tf.concat([gru_output[0], gru_output[1]], -1)

    def get_full_connected_layer(self, layer):
        hidden_units = self.config.hidden_layers[-1]['units']
        output_weight = self.__get_variable([hidden_units, self.label_count], name='output_weight')
        output_bias = tf.Variable(tf.zeros([1, 1, self.label_count]), name='output_bias')
        self.params['output_weight'] = output_weight
        self.params['output_bias'] = output_bias
        return tf.tensordot(layer, output_weight, [[2], [0]]) + output_bias

    def get_dropout_layer(self, layer):
        return tf.layers.dropout(layer, self.config.dropout_rate)

    def __get_variable(self, size, name):
        if name == 'embedding':
            return tf.Variable(tf.random_uniform(size, -0.05, -0.05), name=name)
        else:
            return tf.Variable(tf.truncated_normal(size, stddev=1 / math.sqrt(size[-1])), name=name)
