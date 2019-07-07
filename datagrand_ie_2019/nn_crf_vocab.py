# -*- coding: utf-8 -*-
from pysenal.io import read_lines
from .utils.constant import BATCH_PAD, UNK, BOS, EOS


class NeuralSeqVocab(object):
    def __init__(self, *, dict_path, label_schema):
        self.__dict_path = dict_path
        self.__dictionary = self.__load_dictionary()
        self.__dict_size = len(self.__dictionary)

        self.__label_schema = label_schema
        self.__label_mapping = self.__get_label_mapping()
        self.__reversed_label_mapping = dict(zip(self.__label_mapping.values(), self.__label_mapping.keys()))

    def __load_dictionary(self):
        dictionary = {}

        tokens = read_lines(self.__dict_path, skip_empty=True)
        for idx, token in tokens:
            dictionary[token] = idx
        if BATCH_PAD not in dictionary:
            raise ValueError('PAD is not existed')
        if BOS not in dictionary:
            raise ValueError('BOS is not existed')
        if EOS not in dictionary:
            raise ValueError('EOS is not existed')
        if UNK not in dictionary:
            raise ValueError('UNK is not existed')
        return dictionary

    @property
    def dictionary(self):
        return self.__dictionary

    @property
    def dict_size(self):
        return self.__dict_size

    def __get_label_mapping(self):
        label_mapping = {}
        for index, label in enumerate(self.__label_schema):
            label_mapping[label] = index

        return label_mapping

    @property
    def label_mapping(self):
        return self.__label_mapping

    @property
    def reversed_label_mapping(self):
        return self.__reversed_label_mapping