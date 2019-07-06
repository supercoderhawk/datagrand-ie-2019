# -*- coding: UTF-8 -*-
from pysenal.io import *
from datagrand_ie_2019.utils.constant import *
from datagrand_ie_2019.data_process.label2entity import label2entity


def read_conll_file(conll_filename,
                    conll_delimiter=' ',
                    start_row=0,
                    column_names=('text', 'pos_tag'),
                    column_types=None,
                    text_delimiter=' '):
    """
    read CoNLL file, detect column automatically
    **tokens in CoNLL aren't equal to conll delimiter**
    :param conll_filename: conll filename
    :param conll_delimiter: delimiter of column
    :param start_row: start row index of CoNLL file
    :param column_names: key names in token object, return in columns if column_names is empty
    :param column_types: value type in columns, default is str if column_types is None
    :param text_delimiter: delimiter of tokens, whitespace for EN and empty for CN
    :return: parsed data
    """
    if start_row < 0:
        raise Exception('start row index is less than zero')
    text = '\n'.join(read_lines(conll_filename, skip_empty=False, strip=False)[start_row:])
    sents = [sent for sent in text.split('\n\n') if sent]
    if not sents:
        raise Exception('conll file empty')

    test_item = sents[0].splitlines(False)[0]
    if conll_delimiter not in test_item:
        raise Exception('delimiter error')

    column_count = len(test_item.split(conll_delimiter))
    if column_names and len(column_names) + 1 != column_count:
        raise ValueError('column names count error')
    if column_types:
        if len(column_types) + 1 != column_count:
            raise ValueError('column types count error')
        else:
            for t in column_types:
                if t not in {int, float, str}:
                    raise TypeError('column type {0} is not allowed.'.format(t))

    seq_tags, entity_types = get_tag_sets(conll_filename, conll_delimiter)
    data = [[] for _ in range(column_count)]  # returned data when token_keys is empty
    # returned data when token_keys is not empty
    sent_data = []
    for sent in sents:
        if column_names:
            sent_tokens = []
            sent_labels = []
            for line in sent.splitlines():
                items = line.split(conll_delimiter)
                sent_labels.append(items[-1])
                token = {}
                for item_idx, item in enumerate(items[:-1]):
                    if column_types:
                        item = column_types[item_idx](item)
                    token[column_names[item_idx]] = item
                sent_tokens.append(token)
            sent_text = text_delimiter.join([t[column_names[0]] for t in sent_tokens])
            start = 0
            for token in sent_tokens:
                token['start'] = start
                token['end'] = start + len(token[column_names[0]])
                start = token['end'] + len(text_delimiter)
            entities = label2entity(sent_text, sent_tokens, sent_labels, seq_tags)
            sent_dict = {'text': sent_text, 'tokens': sent_tokens,
                         'labels': sent_labels, 'entities': entities}
            sent_data.append(sent_dict)
        else:
            sent_columns = zip(*[line.split(conll_delimiter) for line in sent.splitlines(False)])
            for col_index, column in enumerate(sent_columns):
                data[col_index].append(column)
    if column_names:
        return sent_data
    else:
        return data


def conll2json(conll_filename,
               json_filename,
               conll_delimiter=' ',
               column_names=('text', 'pos_tag'),
               text_delimiter=' '):
    sents = read_conll_file(conll_filename,
                            conll_delimiter,
                            column_names=column_names,
                            text_delimiter=text_delimiter)
    write_json(json_filename, sents)


def get_tag_sets(conll_filename, delimiter=' ',check_entity_type=False):
    tag_sets = set()
    segments = read_file(conll_filename).split('\n\n')
    for segment in segments:
        for line in segment.splitlines():
            tag_sets.add(line.split(delimiter)[-1])
    seq_tag_type = None
    seq_tags = set()
    type_suffixes = set()

    for tag in tag_sets:
        if tag != 'O':
            if check_entity_type:
                if '-' not in tag:
                    raise ValueError('tag must include type info')
                else:
                    seq_tag, type_suffix = tag.split('-')
                    type_suffixes.add(type_suffix)
            else:
                seq_tag = tag[0]
            seq_tags.add(seq_tag)
        else:
            seq_tags.add(tag)

    if seq_tags == {'B', 'I', 'O'}:
        seq_tag_type = SEQ_BIO
    elif seq_tags == {'B', 'I', 'L', 'O', 'U'}:
        seq_tag_type = SEQ_BILOU
    elif seq_tags != {'B', 'I', 'O'} or seq_tags != {'B', 'I', 'L', 'O', 'U'}:
        raise ValueError('tags are not BIO or BILOU.')

    return seq_tag_type, list(type_suffixes)
