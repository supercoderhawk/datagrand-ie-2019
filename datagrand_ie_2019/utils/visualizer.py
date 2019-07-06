# -*- coding: UTF-8 -*-
"""
highlight entity related functions.
**only work in ipython environment. (ipython console and jupyter notebook)**
"""
from collections import OrderedDict
from pysenal.io import read_json
from .constant import ENTITY_TYPES, DEFAULT_COLOR, EVALUATION_DIR, RESULT_MORE, RESULT_ERROR, RESULT_MISSING
from .utils import *
from .exception import LengthNotEqualException

HIGHLIGHT_COLOR = OrderedDict([
    ('grey', '0;35;47m'),
    ('yellow', '0;30;43m'),
    ('blue', '0;30;44m'),
    ('black', '3;37;40m'),
    ('green', '0;37;42m'),
    ('red', '1;31;40m'),
    ('purple', '0:30:45m')
])


def get_entity_type_color_mapper(entity_types=ENTITY_TYPES):
    if len(entity_types) > len(HIGHLIGHT_COLOR):
        raise ValueError('entity type count is larger than upper bound.')
    color_mapper = {}
    for entity_type, color_name in zip(entity_types, HIGHLIGHT_COLOR):
        color_mapper[entity_type] = color_name
    return color_mapper


class NerVisualizer(object):
    __sep = '=============================='
    __minor_sep = '------------------------------'
    __type_color_mapper = get_entity_type_color_mapper()

    def __init__(self, true_data, *, true_file_suffix='validation', color_mapper=None):
        if isinstance(true_data, str):
            if true_data.endswith(('test_oov.json', 'test.json')):
                true_file_suffix = 'test'
            elif true_data.endswith(('validation_oov.json', 'validation.json')):
                true_file_suffix = 'validation'
        self.__true_file_suffix = true_file_suffix
        self.__true_data = self.__get_data(true_data)
        if color_mapper:
            self.__type_color_mapper = color_mapper

    @classmethod
    def visualize(cls, data, color_mapper=None):
        data = cls.__get_data(data)
        for sent in data:
            entities = copy.deepcopy(sent['entities'])
            cls.__map_color(entities, color_mapper)
            print(cls.__sep)
            print(visualize_spans(sent['text'], entities))

    def compare_model(self, compared_model_name, base_model_name=None, mode=RESULT_ERROR,
                      visualize_true_data=False, evaluation_dir=EVALUATION_DIR):
        compared_filename = self.__get_filename_by_model_name(compared_model_name,
                                                              self.__true_file_suffix,
                                                              evaluation_dir)
        if base_model_name:
            base_filename = self.__get_filename_by_model_name(base_model_name,
                                                              self.__true_file_suffix,
                                                              evaluation_dir)
        else:
            base_filename = None

        if mode == RESULT_MORE:
            if not base_filename:
                raise ValueError('mode **more** must assign base_model_name')
            self.compare_sents_more(compared_filename, base_filename, visualize_true_data)
        elif mode == RESULT_ERROR:
            self.compare_sents_error(compared_filename, base_filename, visualize_true_data)
        elif mode == RESULT_MISSING:
            self.compare_sents_missing(compared_filename, base_filename, visualize_true_data)
        else:
            raise ValueError('invalid comparison mode')

    def compare_sents_more(self, compared_data, base_data, visualize_true_data=False):
        compared_data = self.__get_data(compared_data)
        base_data = self.__get_data(base_data)
        self.__check_input(compared_data, base_data)
        zipped_generator = zip(compared_data, self.__true_data, base_data)
        for compared_sent, true_sent, base_sent in zipped_generator:
            text = compared_sent['text']
            compared_entities = compared_sent['entities']
            true_entities = true_sent['entities']
            base_entities = base_sent['entities']
            more_entities = self.__compare_entity_more(compared_entities, true_entities, base_entities)
            if visualize_true_data:
                self.__simple_visualization(text, more_entities, true_entities)
            else:
                self.__simple_visualization(text, more_entities)

    def compare_sents_missing(self, compared_data, base_data=None, visualize_true_data=False):
        compared_data = self.__get_data(compared_data)
        base_data = self.__get_data(base_data)
        self.__check_input(compared_data, base_data)
        if not base_data:
            base_data = [{'entities': None}] * len(compared_data)
        zipped_items = zip(self.__true_data, compared_data, base_data)
        for true_sent, compared_sent, base_sent in zipped_items:
            text = true_sent['text']
            true_entities = true_sent['entities']
            compared_entities = compared_sent['entities']
            base_entities = base_sent['entities']
            missing_entities = self.__compare_sent_missing(compared_entities,
                                                           true_entities, base_entities)
            if visualize_true_data:
                self.__simple_visualization(text, missing_entities, true_entities)
            else:
                self.__simple_visualization(text, missing_entities)

    def compare_sents_error(self, compared_data, base_data=None, visualize_true_data=False):
        compared_data = self.__get_data(compared_data)
        base_data = self.__get_data(base_data)
        self.__check_input(compared_data, base_data)
        if not base_data:
            base_data = [{'entities': None}] * len(compared_data)
        zipped_items = zip(self.__true_data, compared_data, base_data)
        for true_sent, compared_sent, base_sent in zipped_items:
            text = true_sent['text']
            true_entities = true_sent['entities']
            compared_entities = compared_sent['entities']
            base_entities = base_sent['entities']
            error_entities = self.__compare_sent_error(compared_entities,
                                                       true_entities, base_entities)
            if visualize_true_data:
                self.__simple_visualization(text, error_entities, true_entities)
            else:
                self.__simple_visualization(text, error_entities)

    def __compare_entity_more(self, compared_entities, true_entities, base_entities):
        more_entity_set = EntitySet(compared_entities) - base_entities & true_entities
        more_entities = more_entity_set.entities
        return more_entities

    def __compare_sent_missing(self, compared_entities, true_entities, base_entities=None):
        if not base_entities:
            missing_entity_set = EntitySet(true_entities) - compared_entities
        else:
            missing_entity_set = EntitySet(true_entities) & (EntitySet(base_entities) - compared_entities)
        return missing_entity_set.entities

    def __compare_sent_error(self, compared_entities, true_entities, base_entities=None):
        if not base_entities:
            error_entity_set = EntitySet(compared_entities) - true_entities
        else:
            error_entity_set = EntitySet(compared_entities) - base_entities - true_entities
        return error_entity_set.entities

    @classmethod
    def __map_color(cls, spans, mapper=None):
        if not mapper:
            mapper = cls.__type_color_mapper
        for s in spans:
            e_type = s['type']
            if e_type not in mapper:
                raise KeyError(e_type)
            s['color'] = mapper[e_type]
        return spans

    def __check_input(self, compared_data, base_data):
        self.__data_checker(compared_data)
        if base_data:
            self.__data_checker(base_data)

    def __data_checker(self, data):
        if not isinstance(data, list):
            raise TypeError('input data must be list')
        if len(data) != len(self.__true_data):
            raise LengthNotEqualException('input length and true length is not equal')
        for true_sent, sent in zip(self.__true_data, data):
            if true_sent['text'] != sent['text']:
                raise ValueError('true text and prediction text is not equal.')

    def __simple_visualization(self, text, entities, true_entities=None):
        self.__map_color(entities, self.__type_color_mapper)
        print(self.__sep)
        print(visualize_spans(text, entities))
        if true_entities is not None:
            true_entities = copy.deepcopy(true_entities)
            self.__map_color(true_entities, self.__type_color_mapper)
            print(self.__minor_sep)
            print(visualize_spans(text, true_entities))

    @classmethod
    def __get_data(cls, data):
        if data is None:
            return data
        elif isinstance(data, str):
            return read_json(data)
        elif isinstance(data, list):
            return data
        else:
            raise TypeError('input data type {} is invalid in visualization'.format(type(data)))

    @classmethod
    def __get_filename_by_model_name(cls, model_name, suffix, dirname):
        return os.path.join(dirname, model_name) + '_' + suffix + '.json'


def visualize_spans(text, spans, default_color=DEFAULT_COLOR):
    """
    visualize sentence by spans
    :param text: sentence text to be highlighted
    :param spans: spans list, item is dict, start and end field must be assigned, color is optional.
                    If no color field, default color will be used.
    :param default_color: default color to visualize spans
    :return: visualized text to print
    """
    spans = merge_spans(copy.deepcopy(spans))
    if not spans:
        return text

    display_sentence = text[:spans[0]['start']]
    next_start = [span['start'] for span in spans[1:]] + [len(text)]
    for span, next_s in zip(spans, next_start):
        start = span['start']
        end = span['end']
        color = span.get('color') or default_color

        if end < len(text):
            display_sentence += highlight(text[start:end], color) + text[end:next_s]
        else:
            display_sentence += highlight(text[start:end], color)

    return display_sentence


def visualize_spans_with_token_separator(text, tokens, spans, separator=' ',
                                         default_color=DEFAULT_COLOR):
    """
    highlight tokens by spans, used for non whitespace split language, such as Chinese, Japanese
    :param text: text
    :param tokens: token list
    :param spans: spans list, item is dict, start and end field must be assigned, color is optional.
                    If no color field, default color will be used.
    :param separator: separator to be inserted into text
    :param default_color: default color name
    :return: visualized text to print
    """
    spans = merge_spans(spans)
    tokens = merge_spans(tokens)
    if not spans:
        return ' '.join([token['text'] for token in tokens])

    new_text, char_mapper = insert_separator_in_tokenized_text(text, tokens, separator)
    for span in spans:
        span['start'] = char_mapper[span['start']]
        span['end'] = char_mapper[span['end'] - 1] + 1

    return visualize_spans(new_text, spans, default_color)


def insert_separator_in_tokenized_text(text, tokens, new_suffix):
    if not tokens:
        return tokens
    new_text = text[:tokens[0]['start']]
    offset = 0
    char_mapper = {}
    suffix_len = len(new_suffix)
    last_end = 0

    for token in tokens:
        for idx in range(last_end, token['end']):
            char_mapper[idx] = idx + offset
        new_text += text[last_end:token['end']] + new_suffix
        last_end = token['end']
        offset += suffix_len

    # remove last suffix
    new_text = new_text[:-suffix_len]

    new_text += text[tokens[-1]['end']:]
    for idx in range(tokens[-1]['end'], len(text)):
        char_mapper[idx] = idx + offset

    return new_text, char_mapper


def highlight(s, color):
    """
    add background color of text to highlight it
    :param s: text to highlight
    :param color: color name, must be in `HIGHLIGHT_COLOR dict`
    :return: highlighted text, only work in ipython environment
    """
    return "\033[" + HIGHLIGHT_COLOR[color] + s + "\033[0m"


class EntitySet(object):
    def __init__(self, entities):
        entities, typed_spans = self.__process_input(entities)
        self.__entities = entities
        self.__typed_spans = typed_spans

    def intersection(self, entities):
        entities, typed_spans = self.__process_input(entities)
        intersect_entities = []
        for e in entities:
            span = (e['start'], e['end'], e['type'])
            if span not in self.__typed_spans:
                intersect_entities.append(e)
        return EntitySet(intersect_entities)

    def union(self, entities):
        entities, typed_spans = self.__process_input(entities)
        union_entities = copy.deepcopy(self.__entities)
        for e in entities:
            span = (e['start'], e['end'], e['type'])
            if span not in self.__typed_spans:
                union_entities.append(e)
        return EntitySet(union_entities)

    def difference(self, entities):
        entities, typed_spans = self.__process_input(entities)
        diff_entities = []
        for e in self.__entities:
            span = (e['start'], e['end'], e['type'])
            if span not in typed_spans:
                diff_entities.append(e)
        return EntitySet(diff_entities)

    def __dedupe(self, entities):
        typed_spans = set()
        entity_list = []
        for e in entities:
            span = (e['start'], e['end'], e['type'])
            if span not in typed_spans:
                typed_spans.add(span)
                entity_list.append(e)
        return entity_list, typed_spans

    def __process_input(self, entities):
        if isinstance(entities, EntitySet):
            return entities.entities, entities.typed_spans
        else:
            return self.__dedupe(entities)

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __iter__(self):
        for e in self.__entities:
            yield e

    def __next__(self):
        pass

    @property
    def entities(self):
        return self.__entities

    @property
    def typed_spans(self):
        return self.__typed_spans
