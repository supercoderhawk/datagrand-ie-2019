# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
from datagrand_ie_2019.bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers
from datagrand_ie_2019.utils.constant import DATA_DIR


class ELMO(object):
    options_file = os.path.join(DATA_DIR, 'options.json')
    weight_file = os.path.join(DATA_DIR, 'elmo.hdf5')
    token_embedding_file = os.path.join(DATA_DIR, 'elmo_embedding.hdf5')
    vocab_file = os.path.join(DATA_DIR, 'corpus_vocab.txt')

    def __init__(self, sess, token_placeholder):
        self.bilm = BidirectionalLanguageModel(
            self.options_file,
            self.weight_file,
            use_character_inputs=False,
            embedding_weight_file=self.token_embedding_file
        )
        self.batcher = TokenBatcher(self.vocab_file)
        self.token_placeholder = token_placeholder
        embeddings_op = self.bilm(self.token_placeholder)

        self.elmo_input = weight_layers('input', embeddings_op, l2_coef=0.0)
        self.elmo_output = weight_layers('output', embeddings_op, l2_coef=0.0)

        self.sess = sess

    def run(self, sents):
        token_ids = self.batcher.batch_sentences(sents)
        output = self.sess.run(
            self.elmo_input['weighted_op'],
            feed_dict={self.token_placeholder: token_ids}
        )
        return output
