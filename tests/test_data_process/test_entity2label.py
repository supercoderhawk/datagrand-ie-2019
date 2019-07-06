# -*- coding: UTF-8 -*-
import copy
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.data_process.entity2label import (Entity2Label,
                                                                                     get_index_span,
                                                                                     split_conflict_spans)


def test_entity2label():
    sent = {'text': 'Steve Jobs founded Apple.Inc.',
            'tokens': [{'end': 5, 'start': 0, 'text': 'Steve'},
                       {'end': 10, 'start': 6, 'text': 'Jobs'},
                       {'end': 18, 'start': 11, 'text': 'founded'},
                       {'end': 24, 'start': 19, 'text': 'Apple'},
                       {'end': 25, 'start': 24, 'text': '.'},
                       {'end': 28, 'start': 25, 'text': 'Inc'},
                       {'end': 29, 'start': 28, 'text': '.'}],
            'entities': [{'end': 10, 'entity': 'Steve Jobs', 'start': 0, 'type': 'PER'},
                         {'end': 28, 'entity': 'Apple.Inc', 'start': 19, 'type': 'COMP'}]
            }
    entity2label_bilou = Entity2Label(SEQ_BILOU)
    new_sent = copy.deepcopy(sent)
    entity2label_bilou.sent(new_sent)
    assert new_sent['labels'] == ['B-PER', 'L-PER', 'O', 'B-COMP', 'I-COMP', 'L-COMP', 'O']
    assert new_sent['text'] == sent['text']
    assert new_sent['tokens'] == sent['tokens']
    assert new_sent['entities'] == sent['entities']

    entity2label_bio = Entity2Label(SEQ_BIO)
    new_sent_bio = copy.deepcopy(sent)
    entity2label_bio.sent(new_sent_bio)
    assert new_sent_bio['labels'] == ['B-PER', 'I-PER', 'O', 'B-COMP', 'I-COMP', 'I-COMP', 'O']

    entity2label_single = Entity2Label(None)
    new_sent_single = copy.deepcopy(sent)
    entity2label_single.sent(new_sent_single)
    assert new_sent_single['labels'] == ['PER', 'PER', 'O', 'COMP', 'COMP', 'COMP', 'O']


def test_entity2label_conflict():
    sent = {'text': 'Patsnap.com is a useful website.',
            'tokens': [{'end': 11, 'pos_tag': 'NNP', 'start': 0, 'text': 'Patsnap.com'},
                       {'end': 14, 'pos_tag': 'VBZ', 'start': 12, 'text': 'is'},
                       {'end': 16, 'pos_tag': 'DT', 'start': 15, 'text': 'a'},
                       {'end': 23, 'pos_tag': 'JJ', 'start': 17, 'text': 'useful'},
                       {'end': 31, 'pos_tag': 'NN', 'start': 24, 'text': 'website'},
                       {'end': 32, 'pos_tag': '.', 'start': 31, 'text': '.'}],
            'entities': [{'entity': 'Patsnap', 'start': 0, 'end': 7, 'type': 'Company'}]
            }
    entity2label = Entity2Label(SEQ_BILOU)
    new_sent = copy.deepcopy(sent)
    entity2label.sent(new_sent)
    assert new_sent['labels'] == ['U-Company', 'O', 'O', 'O', 'O', 'O', 'O']
    new_tokens = [{'text': 'Patsnap', 'pos_tag': 'NNP', 'start': 0, 'end': 7},
                  {'text': '.com', 'pos_tag': 'NNP', 'start': 7, 'end': 11},
                  {'text': 'is', 'pos_tag': 'VBZ', 'start': 12, 'end': 14},
                  {'text': 'a', 'pos_tag': 'DT', 'start': 15, 'end': 16},
                  {'text': 'useful', 'pos_tag': 'JJ', 'start': 17, 'end': 23},
                  {'text': 'website', 'pos_tag': 'NN', 'start': 24, 'end': 31},
                  {'text': '.', 'pos_tag': '.', 'start': 31, 'end': 32}]
    assert new_sent['tokens'] == new_tokens
