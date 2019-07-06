# -*- coding: UTF-8 -*-
import pytest
from datagrand_ie_2019.data_process.label2entity import *


def test_label2entity():
    text = 'Steve Jobs founded Apple.Inc.'
    tokens = [{'end': 5, 'start': 0, 'text': 'Steve'},
              {'end': 10, 'start': 6, 'text': 'Job'},
              {'end': 18, 'start': 11, 'text': 'founded'},
              {'end': 24, 'start': 19, 'text': 'Apple'},
              {'end': 25, 'start': 24, 'text': '.'},
              {'end': 28, 'start': 25, 'text': 'Inc'},
              {'end': 29, 'start': 28, 'text': '.'}]
    labels = ['B-PER', 'L-PER', 'O', 'B-COMP', 'I-COMP', 'L-COMP', 'O']
    true_entities = [{'end': 10, 'entity': 'Steve Jobs', 'start': 0, 'type': 'PER'},
                     {'end': 28, 'entity': 'Apple.Inc', 'start': 19, 'type': 'COMP'}]
    entities = label2entity(text, tokens, labels)
    assert entities == true_entities

    with pytest.raises(LengthNotEqualException):
        label2entity(text, tokens, labels[:-1])


def test_label2span_bilou():
    # correct labels
    label = ['O', 'B-IUPAC', 'L-IUPAC', 'O', 'U-IUPAC', 'B-IUPAC', 'I-IUPAC', 'L-IUPAC']
    assert label2span_bilou(label) == [(1, 3, 'IUPAC'), (4, 5, 'IUPAC'), (5, 8, 'IUPAC')]

    # error labels
    label1 = ['B-PPL', 'B-PPL', 'I-PPL', 'L-PPL', 'O']
    assert label2span_bilou(label1) == [(0, 1, 'PPL'), (1, 4, 'PPL')]
    label2 = ['B-PPL', 'I-PPL', 'B-PPL', 'I-PPL', 'O']
    assert label2span_bilou(label2) == [(0, 2, 'PPL'), (2, 4, 'PPL')]

    label3 = ['B-PER', 'I-PER', 'I-PER']
    assert label2span_bilou(label3) == [(0, 3, 'PER')]


def test_label2span_bio():
    # correct labels
    label = ['B-PER', 'I-PER', 'O']
    assert label2span_bio(label) == [(0, 2, 'PER')]
    label2 = ['O', 'O', 'B-PER', 'B-PER', 'O', 'B-COMP', 'I-COMP']
    assert label2span_bio(label2) == [(2, 3, 'PER'), (3, 4, 'PER'), (5, 7, 'COMP')]

    # error labels
    label3 = ['I-PER', 'B-LOC', 'O']
    assert label2span_bio(label3) == [(0, 1, 'PER'), (1, 2, 'LOC')]


def test_check_label_names():
    assert check_label_names(['B', 'I'], SEQ_BIO)
    with pytest.raises(LabelError):
        check_label_names(['I'], SEQ_BIO)

    with pytest.raises(LabelError):
        check_label_names(['O', 'I', 'O'], SEQ_BIO)

    with pytest.raises(LabelError):
        check_label_names(['S', 'K'], SEQ_BIO)

    with pytest.raises(LabelError):
        check_label_names(['B-', 'I-'], SEQ_BIO)
    with pytest.raises(LabelError):
        check_label_names(['B-PER', 'I-PER', 'L-PER'], SEQ_BIO)

    assert check_label_names(['B-PER', 'I-PER', 'L-PER'], SEQ_BILOU)
    assert check_label_names(['O', 'B-LOC', 'L-LOC', 'O'], SEQ_BILOU)
    assert check_label_names(['B', 'L', 'U', 'B', 'I', 'L'], SEQ_BILOU)
    assert check_label_names(['O', 'U', 'B-LOC', 'L-LOC', 'U', 'O'], SEQ_BILOU)
    assert check_label_names(['B', 'L', 'O', 'B', 'I', 'L'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['I', 'O'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['L', 'I'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['O', 'I', 'L'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['B', 'O'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['B', 'L', 'I'], SEQ_BILOU)

    with pytest.raises(LabelError):
        check_label_names(['B', 'L', 'L'], SEQ_BILOU)


def test_get_type_in_labels():
    labels = ['B-PER', 'L-PER']
    assert get_type_in_labels(labels) == 'PER'
    assert get_type_in_labels(['B']) == '{{default_entity_type}}'

    label1 = ['B-PER', 'L-PPL']
    assert get_type_in_labels(label1) == 'PPL'
