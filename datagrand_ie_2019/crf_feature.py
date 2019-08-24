# -*- coding: UTF-8 -*-
"""encapsulate CRF features"""


class CRFFeature(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_texts = [token['text'] for token in tokens]
        self.length = len(self.tokens)

    def word2feature(self, idx):
        all_features = {
            'UNIGRAM': self.token_texts[idx],
            'is_start': idx == 0,
            'is_end': idx == self.length - 1
        }
        if idx:
            all_features['BIGRAM'] = ' '.join(self.token_texts[idx - 1:idx + 1])
            all_features['UNIGRAM:-1'] = self.token_texts[idx - 1]

        if idx > 1:
            all_features['TRIGRAM'] = ' '.join(self.token_texts[idx - 2:idx + 1])
            all_features['context:-2/-1'] = ' '.join(self.token_texts[idx - 2:idx])
            all_features['context:-2'] = self.token_texts[idx - 2]

        if idx > 2:
            all_features['FOURGRAM'] = ' '.join(self.token_texts[idx - 3:idx + 1])
            all_features['context:-3/-2/-1'] = ' '.join(self.token_texts[idx - 3:idx])
            all_features['context:-3/-2'] = ' '.join(self.token_texts[idx - 3:idx - 1])
            all_features['context:-3'] = self.token_texts[idx - 3]

        if idx > 3:
            # all_features['FIVEGRAM'] = ' '.join(self.token_texts[idx - 4:idx + 1])
            # all_features['context:-4/-3/-2/-1'] = ' '.join(self.token_texts[idx - 4:idx])
            # all_features['context:-4/-3'] = ' '.join(self.token_texts[idx - 4:idx - 2])
            all_features['context:-4/-3/-2'] = ' '.join(self.token_texts[idx - 4:idx - 1])
            # all_features['context:-4/-3'] = ' '.join(self.token_texts[idx - 4:idx - 2])

        # if idx > 4:
        # all_features['context:-5/-4'] = ' '.join(self.token_texts[idx - 5:idx - 3])
        # all_features['context:-5/-4/-3'] = ' '.join(self.token_texts[idx - 5:idx - 2])

        if idx < self.length - 1:
            all_features['UNIGRAM:1'] = self.token_texts[idx + 1]
            all_features['BIGRAM:0/1'] = ' '.join(self.token_texts[idx:idx + 2])

        if idx < self.length - 2:
            all_features['TRIGRAM:0/1/2'] = ' '.join(self.token_texts[idx:idx + 3])
            all_features['context:1/2'] = ' '.join(self.token_texts[idx + 1:idx + 3])
            all_features['context:2'] = self.token_texts[idx + 2]

        if idx < self.length - 3:
            all_features['FOUR:NEXT'] = ' '.join(self.token_texts[idx:idx + 4])
            all_features['context:1/2/3'] = ' '.join(self.token_texts[idx + 1:idx + 4])
            all_features['context:2/3'] = ' '.join(self.token_texts[idx + 2:idx + 4])
            all_features['context:3'] = self.token_texts[idx + 3]

        if idx < self.length - 4:
            # all_features['FIVEGRAM:NEXT'] = ' '.join(self.token_texts[idx:idx + 5])
            # all_features['context:1/2/3/4'] = ' '.join(self.token_texts[idx + 1:idx + 5])
            all_features['context:2/3/4'] = ' '.join(self.token_texts[idx + 2:idx + 5])
            # all_features['context:3/4'] = ' '.join(self.token_texts[idx + 3:idx + 5])
            # all_features['context:3/4'] = ' '.join(self.token_texts[idx + 3:idx + 5])

        # if idx < self.length - 5:
        # all_features['context:3/4/5'] = ' '.join(self.token_texts[idx + 3:idx + 6])
        # all_features['context:4/5'] = ' '.join(self.token_texts[idx + 3:idx + 6])

        if 1 <= idx < self.length - 1:
            all_features['TRIGRAM:-1/0/1'] = ' '.join(self.token_texts[idx - 1:idx + 2])
            all_features['context:-1/1'] = self.get_context_feature(idx - 1, idx, idx + 1, idx + 2)

        if 2 <= idx < self.length - 2:
            # all_features['FIVEGRAM:-2/-1/0/1/2'] = ' '.join(self.token_texts[idx - 2:idx + 3])
            all_features['context:-2/-1/1/2'] = self.get_context_feature(idx - 2, idx, idx + 1, idx + 3)
            all_features['context:-2/1/2'] = self.get_context_feature(idx - 2, idx - 1, idx + 1, idx + 3)
            all_features['context:-2/-1/2'] = self.get_context_feature(idx - 2, idx, idx + 2, idx + 3)

        # if 3 <= idx < self.length - 3:
        #     all_features['context:-3/-2/2/3'] = self.get_context_feature(idx - 3, idx - 1, idx + 2, idx + 4)
        #     all_features['context:-3/-2/2'] = self.get_context_feature(idx - 3, idx - 1, idx + 2, idx + 3)
        #     all_features['context:-2/2/3'] = self.get_context_feature(idx - 2, idx - 1, idx + 2, idx + 4)

        return all_features

    def sent2feature(self):
        return (self.word2feature(i) for i in range(self.length))

    def get_context_feature(self, prev_start, prev_end, next_start, next_end):
        prev_text = ' '.join(self.token_texts[prev_start:prev_end])
        next_text = ' '.join(self.token_texts[next_start:next_end])
        feature = prev_text + ' - ' + next_text
        return feature
