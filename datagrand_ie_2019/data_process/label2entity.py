# -*- coding: UTF-8 -*-
"""
transform labels to entities, also provide labels interaction validation method
"""
import re
from ..utils.exception import LengthNotEqualException, LabelError
from ..utils.constant import DEFAULT_TYPE, SEQ_BILOU, SEQ_BIO


def label2entity(tokens, labels, label_schema=SEQ_BILOU, is_check_labels=False):
    """
    transform labels to entities
    :param text: original text
    :param tokens: token list, every token in dict and has 'text','start' and 'end'
    :param labels: labels correspond to tokens
    :param label_schema: label schema, only allows BIO and BILOU
    :param is_check_labels: whether check labels, default is False
    :return: transformed entity list
    """
    if len(tokens) != len(labels):
        raise LengthNotEqualException('label2entity: token and label count are not equal.')
    if is_check_labels:
        check_label_names(labels, label_schema)

    spans = label2span(labels, label_schema)

    entities = []
    for start_idx, end_idx, entity_type in spans:
        entity = {'start': start_idx,
                  'end': end_idx,
                  'type': entity_type}
        entities.append(entity)

    return entities


def label2span(labels, label_schema):
    """
    transform label to spans according to label schema
    :param labels: labels
    :param label_schema: label schema, only in BIO and BILOU
    :return: spans
    """
    if label_schema == SEQ_BILOU:
        spans = label2span_bilou(labels)
    elif label_schema == SEQ_BIO:
        spans = label2span_bio(labels)
    else:
        raise ValueError('label schema {0} is not supported'.format(label_schema))
    return spans


def label2span_bilou(labels):
    """
    transform labels in BILOU schema to spans, which everyone in (start_index, end_index, entity_type) format.
    For example,
        input ['B-LOC','I-LOC', 'L-LOC', 'O]
        output [(0, 3, 'LOC')]
    :param labels: labels to be transformed
    :return: spans
    """
    if not labels:
        return []
    if type(labels) not in {list, tuple}:
        raise TypeError('labels must be in list or tuple')

    start = -1
    spans = []
    prev_labels = ['S'] + labels[:-1]
    for label_idx, (prev_label, label) in enumerate(zip(prev_labels, labels)):
        if label.startswith('U'):
            if prev_label.startswith(('B', 'I')):
                span = (start, label_idx, get_type_in_labels(labels[start:label_idx]))
                spans.append(span)
            span = (label_idx, label_idx + 1, get_type_in_labels([label]))
            spans.append(span)
        elif label.startswith('B'):
            if prev_label.startswith(('B', 'I')):
                span = (start, label_idx, get_type_in_labels(labels[start:label_idx]))
                spans.append(span)
            start = label_idx
        elif label.startswith('L'):
            if prev_label.startswith(('S', 'O', 'U')):
                start = label_idx
            end = label_idx + 1
            span = (start, end, get_type_in_labels(labels[start:end]))
            spans.append(span)
        elif label.startswith('I'):
            if prev_label.startswith(('S', 'O', 'U')):
                start = label_idx
        elif label.startswith('O') and prev_label.startswith(('B', 'I')):
            span = (start, label_idx, get_type_in_labels(labels[start:label_idx]))
            spans.append(span)
    if labels[-1].startswith('I'):
        span = (start, len(labels), get_type_in_labels(labels[start:]))
        spans.append(span)
    return spans


def label2span_bio(labels):
    """
    transform labels in BIO schema to spans, which everyone in (start_index, end_index, entity_type) format.
    For example,
        input ['B-LOC','I-LOC', 'O]
        output [(0, 2, 'LOC')]
    :param labels: labels to be transformed
    :return: spans
    """
    if not labels:
        return []
    if type(labels) not in {list, tuple}:
        raise TypeError('labels must be in list or tuple')

    start = -1
    spans = []
    prev_labels = ['S'] + labels[:-1]

    for label_idx, (prev_label, label) in enumerate(zip(prev_labels, labels)):
        if label.startswith('B'):
            if prev_label.startswith('B') or prev_label.startswith('I'):
                span = (start, label_idx, get_type_in_labels(labels[start:label_idx]))
                spans.append(span)
            start = label_idx
        elif label.startswith('O'):
            if prev_label.startswith('B') or prev_label.startswith('I'):
                span = (start, label_idx, get_type_in_labels(labels[start:label_idx]))
                spans.append(span)
        elif label.startswith('I'):
            if prev_label.startswith(('S', 'O')):
                start = label_idx

    if not labels[-1].startswith('O'):
        span = (start, len(labels), get_type_in_labels(labels[start:]))
        spans.append(span)
    return spans


def get_type_in_labels(labels):
    """
    get entity type by its all labels
    :param labels: labels represent the single entity
    :return: entity type
    """
    # use type info in last label
    label = labels[-1]
    # compatible with label without entity type info (not recommend)
    if len(label) > 2:
        entity_type = label[2:]
    else:
        entity_type = DEFAULT_TYPE
    return entity_type


__BILOU_INTERACTION = {'S': ('B', 'U', 'O'),
                       'B': ('I', 'L'),
                       'I': ('I', 'L'),
                       'L': ('O', 'B', 'U'),
                       'O': ('B', 'O', 'U'),
                       'U': ('U', 'B', 'O')}

__BIO_INTERACTION = {'S': ('B', 'O'),
                     'B': ('I', 'O'),
                     'I': ('I', 'O', 'B'),
                     'O': ('B', 'O')}

__BIO_LABEL_REGEX = re.compile(r'^[BIO]$|^[BIO]-.+$')
__BILOU_LABEL_REGEX = re.compile(r'^[BILOU]$|^[BILOU]-.+$')


def check_label_names(labels, label_schema, raise_exception=True):
    """
    check label, including label name and labels interaction
    :param labels: labels to be checked
    :param label_schema: label schema, only allow BIO and BILOU
    :param raise_exception: whether raise exception when error occurs
    :return: Whether labels are validated
    """
    if label_schema == SEQ_BIO:
        interaction = __BIO_INTERACTION
        regex = __BIO_LABEL_REGEX
    elif label_schema == SEQ_BILOU:
        interaction = __BILOU_INTERACTION
        regex = __BILOU_LABEL_REGEX
    else:
        raise Exception('error label schema {}'.format(label_schema))

    prev_labels = ['S'] + labels[:-1]
    for idx, (prev_label, label) in enumerate(zip(prev_labels, labels)):
        if not regex.fullmatch(label):
            raise LabelError(name=label, schema=label_schema)
        if label[0] not in interaction[prev_label[0]]:
            if raise_exception:
                raise LabelError(index=idx, label=[prev_label, label])
            else:
                return False
    return True
