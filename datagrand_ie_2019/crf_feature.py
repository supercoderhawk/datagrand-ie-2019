# -*- coding: UTF-8 -*-
"""encapsulate CRF features"""
from .utils.constant import *


class CRFFeature(object):
    feature_sets = []

    def __init__(self, tokens):
        self.tokens = tokens
        self.token_texts = [token['text'] for token in tokens]
        self.length = len(self.tokens)

    def word2feature(self, idx):
        all_features = {
            UNIGRAM: self.token_texts[idx]
        }
        if idx:
            all_features[BIGRAM] = ' '.join(self.token_texts[idx - 1:idx + 1])
            all_features[UNIGRAM_PREV_1] = self.token_texts[idx - 1]
        if idx > 1:
            all_features[TRIGRAM] = ' '.join(self.token_texts[idx - 2:idx + 1])
        if idx > 2:
            all_features[FOURGRAM] = ' '.join(self.token_texts[idx - 3:idx + 1])
        if idx > 3:
            all_features[FIVEGRAM] = ' '.join(self.token_texts[idx - 4:idx + 1])

        if idx < self.length - 1:
            all_features[UNIGRAM_NEXT_1] = self.token_texts[idx + 1]
            all_features[BIGRAM_NEXT] = ' '.join(self.token_texts[idx:idx + 2])

        if idx < self.length - 2:
            all_features[TRIGRAM_NEXT] = self.token_texts[idx:idx + 3]

        if 1 < idx < self.length - 1:
            all_features[TRIGRAM_MID] = ' '.join(self.token_texts[idx - 1:idx + 2])

        if 2 < idx < self.length - 2:
            all_features[FIVEGRAM_MID] = ' '.join(self.token_texts[idx - 2:idx + 3])

        if self.feature_sets:
            features = {feat: feat_val for feat, feat_val in all_features.items() if feat in self.feature_sets}
        else:
            features = all_features
        return features

    def sent2feature(self):
        return (self.word2feature(i) for i in range(self.length))
