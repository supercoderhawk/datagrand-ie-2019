# -*- coding: UTF-8 -*-
"""encapsulate CRF features"""


class CRFFeature(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_texts = [token['text'] for token in tokens]
        self.length = len(self.tokens)

    def word2feature(self, idx):
        all_features = {
            'UNIGRAM': self.token_texts[idx]
        }
        if idx:
            all_features['BIGRAM'] = ' '.join(self.token_texts[idx - 1:idx + 1])
            all_features['UNIGRAM:-1'] = self.token_texts[idx - 1]

        if idx > 1:
            all_features['TRIGRAM'] = ' '.join(self.token_texts[idx - 2:idx + 1])
            all_features['context:-2/-1'] = ' '.join(self.token_texts[idx - 3:idx])
        if idx > 2:
            all_features['FOURGRAM'] = ' '.join(self.token_texts[idx - 3:idx + 1])
            all_features['context:-3/-2/-1'] = ' '.join(self.token_texts[idx - 3:idx])
            all_features['context:-3/-2'] = ' '.join(self.token_texts[idx - 3:idx - 1])
        if idx > 3:
            all_features['FIVEGRAM'] = ' '.join(self.token_texts[idx - 4:idx + 1])
            all_features['context:-4/-3/-2/-1'] = ' '.join(self.token_texts[idx - 4:idx])

        if idx < self.length - 1:
            all_features['UNIGRAM:1'] = self.token_texts[idx + 1]
            all_features['BIGRAM:0/1'] = ' '.join(self.token_texts[idx:idx + 2])

        if idx < self.length - 2:
            all_features['TRIGRAM:0/1/2'] = ' '.join(self.token_texts[idx:idx + 3])
            all_features['context:1/2'] = ' '.join(self.token_texts[idx + 1:idx + 3])

        if idx < self.length - 3:
            all_features['FOUR:NEXT'] = ' '.join(self.token_texts[idx:idx + 4])
            all_features['context:1/2/3'] = ' '.join(self.token_texts[idx + 1:idx + 4])
            all_features['context:2/3'] = ' '.join(self.token_texts[idx + 2:idx + 4])

        if idx < self.length - 4:
            all_features['FIVEGRAM:NEXT'] = ' '.join(self.token_texts[idx:idx + 5])
            all_features['context:1/2/3/4'] = ' '.join(self.token_texts[idx + 1:idx + 5])
            all_features['context:2/3/4'] = ' '.join(self.token_texts[idx + 2:idx + 5])

        if 1 < idx < self.length - 1:
            all_features['TRIGRAM:-1/0/1'] = ' '.join(self.token_texts[idx - 1:idx + 2])
            all_features['context:-1/1'] = self.token_texts[idx - 1] + ' ' + self.token_texts[idx + 1]

        if 2 < idx < self.length - 2:
            all_features['FIVEGRAM:-2/-1/0/1/2'] = ' '.join(self.token_texts[idx - 2:idx + 3])
            four_gram_mid = ' '.join(self.token_texts[idx - 2:idx] + self.token_texts[idx + 1:idx + 3])
            all_features['context:-2/-1/1/2'] = four_gram_mid

        return all_features

    def sent2feature(self):
        return (self.word2feature(i) for i in range(self.length))
