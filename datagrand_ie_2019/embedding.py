# -*- coding: UTF-8 -*-
from glove import Corpus, Glove


def train_glove(src_filename):
    corpus = Corpus()
    corpus.fit(get_lines(src_filename), window=10)
    glove = Glove(no_components=100, learning_rate=0.05)

    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')


def get_lines(src_filename):
    pass


class GloveTrainer(object):
    def __init__(self):
        pass

    def build_corpus(self):
        pass
