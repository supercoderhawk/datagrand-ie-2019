# -*- coding: UTF-8 -*-
"""
define training data loader class
"""
import random
import numpy as np
from pysenal.io import read_json, read_jsonline
from pysenal.utils.logger import get_logger
from .utils.constant import *


class CRFDataLoader(object):
    def __init__(self, filename):
        self.filename = filename
        self.logger = get_logger('[CRFDataLoader]')

    def load_data(self):
        if self.filename.endswith('.json'):
            data = read_json(self.filename)
        elif self.filename.endswith('.jsonl'):
            data = read_jsonline(self.filename)
        else:
            data = read_json(self.filename)
            self.logger.warning('Your file suffix is not json or jsonl')
        return data


class NeuralSeqDataLoader(object):
    """Provides the sequence labeling data that can be feed into neural network from conll file.

    """

    def __init__(self, *, filename, word_mapper, label_mapper, sent_padding_length,
                 is_skip_window=False, skip_left=0, skip_right=0):
        """Creates the sequence labeling data object

        :param filename: conll filename
        :param word_mapper:
        :param label_mapper:
        :param sent_padding_length:
        :param is_skip_window:
        :param skip_left:
        :param skip_right:
        """
        self.__filename = filename
        self.__word2id_mapper = word_mapper
        self.__label2id_mapper = label_mapper
        self.__sent_padding_length = sent_padding_length
        self.__is_skip_window = is_skip_window
        self.__skip_left = skip_left
        self.__skip_right = skip_right
        self.__offset = 0
        self.__sent_count = 0
        self.__sents, self.__labels, self.__seq_lengths = self.__transform()

    def __transform(self):
        sents = []
        labels = []
        seq_lengths = []
        input_sents = read_json(self.__filename)
        random.shuffle(input_sents)
        random.shuffle(input_sents)
        random.shuffle(input_sents)
        random.shuffle(input_sents)
        random.shuffle(input_sents)
        for sent in input_sents:
            sent_words = [t['text'] for t in sent['tokens']]
            sent_labels = sent['labels']
            mapped_words = [self.__word2id_mapper[word] for word in sent_words]
            mapped_labels = [self.__label2id_mapper[label] for label in sent_labels]
            if len(mapped_words) >= self.__sent_padding_length:
                mapped_words = mapped_words[:self.__sent_padding_length]
                mapped_labels = mapped_labels[:self.__sent_padding_length]
            else:
                pad_idx = self.__word2id_mapper[BATCH_PAD]
                mapped_words += [pad_idx] * (self.__sent_padding_length - len(sent_words))
                mapped_labels += [0] * (self.__sent_padding_length - len(sent_labels))
            if self.__is_skip_window:
                sents.append(self.__indices2index_windows(mapped_words))
            else:
                sents.append(mapped_words)
            labels.append(mapped_labels)
            seq_lengths.append(len(sent_labels))
            self.__sent_count += 1
        return np.array(sents), np.array(labels), np.array(seq_lengths)

    def __indices2index_windows(self, seq_indices):
        ext_indices = [self.__word2id_mapper[BOS]] * self.__skip_left
        ext_indices.extend(seq_indices + [self.__word2id_mapper[EOS]] * self.__skip_right)
        seq = []
        for index in range(self.__skip_left, len(ext_indices) - self.__skip_right):
            seq.append(ext_indices[index - self.__skip_left: index + self.__skip_right + 1])

        return seq

    def __get_batch(self, batch_size):
        if self.__offset + batch_size <= self.__sent_count:
            s = slice(self.__offset, self.__offset + batch_size)
            self.__offset += batch_size
            return self.__sents[s], self.__labels[s], self.__seq_lengths[s]
        else:
            s1 = slice(self.__offset, self.__sent_count)
            s2 = slice(0, self.__offset + batch_size - self.__sent_count)
            prefix_sents = self.__sents[s1]
            prefix_labels = self.__labels[s1]
            prefix_lengths = self.__seq_lengths[s1]
            self.__shuffle_data()
            sents = np.concatenate((prefix_sents, self.__sents[s2]))
            labels = np.concatenate((prefix_labels, self.__labels[s2]))
            seq_lengths = np.concatenate((prefix_lengths, self.__seq_lengths[s2]))
            self.__offset += batch_size - self.__sent_count
            return sents, labels, seq_lengths

    def __shuffle_data(self):
        new_indices = list(range(self.__sent_count))
        random.shuffle(new_indices)
        random.shuffle(new_indices)
        random.shuffle(new_indices)
        random.shuffle(new_indices)
        random.shuffle(new_indices)

        self.__sents = self.__sents[new_indices]
        self.__labels = self.__labels[new_indices]
        self.__seq_lengths = self.__seq_lengths[new_indices]

    def mini_batch(self, batch_size):
        while True:
            yield self.__get_batch(batch_size)

    @property
    def sent_count(self):
        return self.__sent_count
