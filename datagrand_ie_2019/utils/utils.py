# -*- coding: UTF-8 -*-
import os
import copy
from collections import defaultdict


def check_entities(entities, text):
    """
    check whether entity object is legal, if error happen, exception will be raised
    :param entities: entity object list
    :param text: original text
    :return: None
    """
    for entity in entities:
        start = entity['start']
        end = entity['end']
        if start < 0 or end < 0:
            raise Exception('offset is negative')
        if text[start:end] != entity['entity']:
            print(entity)
            print(text[start:end])

            raise Exception('entity text and offset don\'t correspond.')


def adjust_entities_offsets(entity_list, offset, start=None, end=None):
    for entity in entity_list:
        not_restrict = not start and not end
        restrict_start = start and start <= entity['start']
        restrict_end = end and end >= entity['end']
        restrict_all = start and end and start <= entity['start'] < entity['end'] <= end
        if not_restrict or restrict_all or restrict_start or restrict_end:
            entity['start'] += offset
            entity['end'] += offset
    return entity_list


def get_filenames_in_folder(folder_name, ext_name=True, hidden_file=False, attach_folder_name=True):
    filenames = []
    if not os.path.exists(folder_name):
        raise Exception('folder is not existed.')
    for filename in os.listdir(folder_name):
        if hidden_file:
            if filename.startswith('.') and filename not in {'.', '..'}:
                filenames.append(filename)
        elif not filename.startswith('.'):
            filenames.append(filename)
    if attach_folder_name:
        filenames = [os.path.join(folder_name, name) for name in filenames]
    if not ext_name:
        filenames = [name[:name.rindex('.')] for name in filenames]
    return filenames


def get_entities_by_type(entity_list, entity_type):
    selected_entities = []

    for entity in entity_list:
        if entity['type'] == entity_type:
            selected_entities.append(entity)

    return selected_entities


def get_index_char2word(tokens, index):
    for idx, token in enumerate(tokens):
        if token['end'] > index:
            return idx
    raise IndexError('index is out of sentence')


def replace_item_in_list(old_list, replaced_items, extend_list=False):
    """
    replace item in list by index
    :param old_list: original list
    :param replaced_items: new items to replace item in list.
                            Every item is in tuple format of `(index, new_item)`
    :param extend_list: whether extend new item when it is list, default is append to list
    :return: new list after replacement
    """
    replaced_items = sorted(replaced_items, key=lambda i: i[0])
    new_list = []
    for idx, item in enumerate(old_list):
        if not replaced_items:
            new_list.extend(old_list[idx:])
            break
        replaced_item = replaced_items[0]
        if idx == replaced_item[0]:
            if extend_list and isinstance(replaced_item[1], list):
                new_list.extend(replaced_item[1])
            else:
                new_list.append(replaced_item[1])
            replaced_items.pop(0)
        else:
            new_list.append(item)
    return new_list


def replace_extname(src_filename, new_extname):
    """
    replace extension name
    :param src_filename: source filename
    :param new_extname: new extension name to replace, dot will be appended automatically if not existed
    :return: new filename
    """
    if not new_extname.startswith('.'):
        new_extname = '.' + new_extname
    return os.path.splitext(src_filename)[0] + new_extname


def split_lines(text, skip_empty=True):
    """
    split text into lines, return lines and spans
    :param text: original text to be split
    :param skip_empty: whether skip empty lines in result
    :return: split lines and line spans
    """
    lines = []
    offset = 0
    for line_text in text.splitlines(keepends=True):
        if skip_empty and line_text.rstrip() or not skip_empty:
            line = {'text': line_text.rstrip(), 'start': offset,
                    'end': offset + len(line_text.rstrip())}
            lines.append(line)

        offset += len(line_text)

    return lines


def merge_spans(spans, merge_adjacency=False):
    """
    merge spans with overlaps, ensure every position is in 1 span at most.
    :param spans: span list
    :param merge_adjacency: whether merge adjacent spans
    :return: merged span list without overlaps
    """
    if not spans:
        return spans

    spans = sorted(spans, key=lambda i: i['start'])
    new_spans = [spans[0]]
    for span in spans[1:]:
        start = span['start']
        end = span['end']
        last_span = new_spans[-1]
        if last_span['end'] < start:
            new_spans.append(span)
        else:
            if not merge_adjacency and last_span['end'] == start:
                new_spans.append(span)
                continue
            if start >= last_span['start'] and end <= last_span['end']:
                continue
            else:
                span = copy.deepcopy(span)
                span.pop('start')
                span.pop('end')
                new_spans[-1] = {'start': last_span['start'], 'end': end}
                new_spans[-1].update(span)

    return new_spans


def entities_to_typed_dict(entities, entity_types=None):
    typed_entity_dict = defaultdict(list)

    for entity in entities:
        typed_entity_dict[entity['type']].append(entity)

    # assign empty list for type which is not occurred in input entity list
    if entity_types:
        for entity_type in entity_types:
            typed_entity_dict.setdefault(entity_type, [])

    return typed_entity_dict


def entities2spans(entities, is_type=False):
    if not is_type:
        spans = [(e['start'], e['end']) for e in entities]
    else:
        spans = [(e['start'], e['end'], e['type']) for e in entities]
    return spans
