# -*- coding: UTF-8 -*-
import random
from pysenal.io import *
from glove import Corpus, Glove
from gensim.models import Word2Vec
from datagrand_ie_2019.utils.constant import *


def build_corpus():
    training_count = 1
    test_count = 1
    all_lines = []
    training_data = read_json(TRAINING_FILE)
    test_data = read_json(TEST_FILE)

    for line in read_lines_lazy(RAW_DATA_DIR + 'corpus.txt'):
        new_line = ' '.join(line.strip().split('_'))
        all_lines.append(new_line)
    for i in range(training_count):
        for sent in training_data:
            new_line = ' '.join([t['text'] for t in sent['tokens']])
            all_lines.append(new_line)
    for i in range(test_count):
        for sent in test_data:
            new_line = ' '.join([t['text'] for t in sent['tokens']])
            all_lines.append(new_line)
    random.shuffle(all_lines)
    random.shuffle(all_lines)
    random.shuffle(all_lines)
    random.shuffle(all_lines)
    random.shuffle(all_lines)
    write_lines(DATA_DIR + 'glove_corpus.txt', all_lines)


def train_glove(src_filename, dim=100):
    corpus = Corpus()
    corpus.fit(get_lines(src_filename), window=10)
    glove = Glove(no_components=dim, learning_rate=0.001)
    glove.fit(corpus.matrix, epochs=100, no_threads=20, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save(DATA_DIR + 'glove.{}d.model'.format(dim))


def get_lines(src_filename):
    for line in read_lines_lazy(src_filename):
        yield line.split(' ')


def train_word2vector(src_filename):
    from gensim.test.utils import common_texts, get_tmpfile
    from gensim.models.word2vec import LineSentence
    # path = get_tmpfile(DATA_DIR + "word2vec.model")
    sents = LineSentence(src_filename)
    model = Word2Vec(sents, size=500, window=10, min_count=1, workers=40)
    # model.train()
    model.save(DATA_DIR + "word2vec.model")


if __name__ == '__main__':
    # build_corpus()
    train_word2vector(DATA_DIR + 'glove_corpus.txt')
    # train_glove(DATA_DIR + 'glove_corpus.txt', 100)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 200)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 300)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 400)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 500)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 600)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 700)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 800)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 900)
    # train_glove(DATA_DIR + 'glove_corpus.txt', 1000)
