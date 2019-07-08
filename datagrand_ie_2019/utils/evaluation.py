# -*- coding: UTF-8 -*-
"""
encapsulate some evaluation related functions
"""
import copy
from collections import OrderedDict
from pysenal.io import read_json
import pandas as pd
from .constant import SEQ_BILOU, ENTITY_TYPES
from .utils import *
from .exception import LengthNotEqualException
from ..data_process.label2entity import label2span


class KFoldEntityEvaluator(object):
    def __init__(self, k, true_filename, entity_types=ENTITY_TYPES):
        self.k = k
        self.__entity_types = entity_types
        evaluators = []
        true_data = read_json(true_filename)
        if k != len(true_data):
            raise ValueError('k and true data does not correspond.')
        for single_fold_true_data in true_data:
            evaluator = EntityEvaluator(single_fold_true_data, entity_types=entity_types)
            evaluators.append(evaluator)
        self.__evaluators = evaluators

    def evaluate(self, pred_filename):
        pred_data = read_json(pred_filename)
        kfold_counter = {}
        for i in range(self.k):
            single_fold_pred_data = pred_data[i]
            ret = self.__evaluators[i].evaluate(single_fold_pred_data, is_percentage=False)
            for metrics, e_counter in ret.items():
                for e_type, val in e_counter.items():
                    kfold_counter[metrics + '-' + e_type][str(i)] = val
        kfold_counter = pd.DataFrame(kfold_counter)
        counter_sum = kfold_counter.sum(axis=1)
        macro_row = {}
        micro_row = {}
        for e in self.__entity_types:
            micro_precision = counter_sum['true_positive_count-' + e] / counter_sum['pred_count-' + e]
            micro_recall = counter_sum['true_positive_count-' + e] / counter_sum['true_count-' + e]
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            micro_row['precision-' + e] = micro_precision
            micro_row['recall-' + e] = micro_recall
            micro_row['f1-' + e] = micro_f1
            macro_row['precision-' + e] = counter_sum['precision-' + e] / self.k
            macro_row['recall-' + e] = counter_sum['recall-' + e] / self.k
            macro_row['f1-' + e] = counter_sum['f1-' + e] / self.k
        kfold_counter.append(macro_row)
        kfold_counter.append(micro_row)
        for e in self.__entity_types:
            col_names = ['true_positive_count-' + e, 'pred_count-' + e]
            kfold_counter.drop(col_names, axis='columns', inplace=True)
        return kfold_counter


class EntityEvaluator(object):
    def __init__(self, true_data, oov_true_data=None, entity_types=ENTITY_TYPES):
        self.true_data = self.__get_data(true_data)
        self.oov_true_data = self.__get_data(oov_true_data)
        self.is_evaluate_oov = self.oov_true_data is not None
        self.entity_types = entity_types

    def evaluate(self, pred_data, is_count=False, is_percentage=True, decimal=2):
        if is_percentage:
            decimal += 2
        pred_data = self.__get_data(pred_data)
        counter = self.evaluate_data(pred_data, decimal)
        if self.is_evaluate_oov:
            oov_counter = self.evaluate_data(pred_data, decimal)
            counter.insert(3, 'oov_rate', oov_counter['recall'])
            counter['oov_count'] = oov_counter['true_count']
        if not is_count:
            counter.drop(['true_positive_count', 'pred_count', 'true_count'],
                         axis='columns', inplace=True)
            if self.is_evaluate_oov:
                counter.drop(['oov_count'], axis='columns', inplace=True)
        if is_percentage:
            self.to_percentage(counter, decimal - 2)
        return counter

    def evaluate_data(self, pred_data, decimal, return_dataframe=True):
        self.__check_data(pred_data)
        counter = {}
        typed_pred_data = aggregate_entities_in_sents_by_type(pred_data, self.entity_types)
        typed_true_data = aggregate_entities_in_sents_by_type(self.true_data, self.entity_types)

        for entity_type in self.entity_types:
            item = self.__evaluate_single_type(entity_type, typed_pred_data, typed_true_data)
            for k, v in item.items():
                counter.setdefault(k, {})
                counter[k][entity_type] = v
        if return_dataframe:
            counter = pd.DataFrame(counter).round(decimal)
        return counter

    def __evaluate_single_type(self, entity_type, pred_data, true_data):
        pred_count = 0
        true_count = 0
        true_positive_count = 0
        pred_data_single_type = pred_data[entity_type]
        true_data_single_type = true_data[entity_type]
        zipped_pred_true_data = zip(pred_data_single_type, true_data_single_type)
        for pred_sent, true_sent in zipped_pred_true_data:
            pred_entities = pred_sent['entities']
            true_entities = true_sent['entities']

            sent_tp_count = self.get_true_positive_count(pred_entities,
                                                         true_entities)
            true_positive_count += sent_tp_count
            pred_count += len(pred_entities)
            true_count += len(true_entities)
        if not pred_count:
            precision = 0
        else:
            precision = true_positive_count / pred_count
        if not true_count:
            recall = 0
        else:
            recall = true_positive_count / true_count
        if not (precision + recall):
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # order dict is used to keep column name in order in pandas DataFrame
        item = OrderedDict([
            ('precision', precision),
            ('recall', recall),
            ('f1', f1),
            ('true_positive_count', true_positive_count),
            ('pred_count', pred_count),
            ('true_count', true_count)
        ])
        return item

    @classmethod
    def get_true_positive_count(cls, pred_entities, true_entities):
        pred_spans = set((e['start'], e['end']) for e in pred_entities)
        true_spans = set((e['start'], e['end']) for e in true_entities)
        true_positive_spans = pred_spans.intersection(true_spans)
        return len(true_positive_spans)

    @classmethod
    def to_percentage(self, dataframe, decimal):
        for idx, column in dataframe.iteritems():
            dataframe[idx] = column.map(lambda i: str(round(i * 100, decimal)) + '%')

    def __get_data(self, data):
        if data is None:
            return data
        elif isinstance(data, str):
            return read_json(data)
        elif isinstance(data, list):
            return data
        else:
            raise TypeError('input data type {} is invalid in evaluation'.format(type(data)))

    def __check_data(self, pred_data):
        if len(pred_data) != len(self.true_data):
            raise LengthNotEqualException('pred data count is not equal to true data count')
        msg_tmpl = 'evaluation data error. {} is not equal to {}'

        for pred_sent, true_sent in zip(pred_data, self.true_data):
            pred_tokens = pred_sent['tokens']
            true_tokens = true_sent['tokens']
            if pred_tokens != true_tokens:
                raise ValueError(msg_tmpl.format(pred_tokens, true_tokens))


class LabelEvaluator(object):
    def __init__(self, true_data, oov_true_data=None,
                 label_schema=SEQ_BILOU, entity_types=ENTITY_TYPES):
        self.__label_schema = label_schema
        self.__entity_types = entity_types
        self.__true_data = self.__get_data(true_data)
        self.__oov_true_data = self.__get_data(oov_true_data)

    def evaluate(self, pred_data):
        self.__check_data(pred_data)
        typed_pred_spans = {}
        typed_true_spans = {}
        # aggregate sents labels to entity typed label
        for pred_sent, true_sent in zip(pred_data, self.__true_data):
            sent_ret = self.__sent_label2span(pred_sent['labels'], true_sent['labels'])
            for entity_type in self.__entity_types:
                typed_pred_spans.setdefault(entity_type, []).extend(sent_ret['pred'][entity_type])
                typed_true_spans.setdefault(entity_type, []).extend(sent_ret['true'][entity_type])

        counter = OrderedDict([
            ('precision', OrderedDict()),
            ('recall', OrderedDict()),
            ('f1', OrderedDict()),
            ('oov_rate', OrderedDict()),
            ('true_positive_count', OrderedDict()),
            ('pred_count', OrderedDict()),
            ('true_count', OrderedDict()),
            ('oov_count', OrderedDict())
        ])

        for entity_type in self.__entity_types:
            pred_spans = typed_pred_spans[entity_type]
            true_spans = typed_true_spans[entity_type]
            joined_spans = set(pred_spans).intersection(true_spans)
            true_positive = len(joined_spans)
            precision = true_positive / len(pred_spans)
            recall = true_positive / len(true_spans)
            counter['precision'][entity_type] = precision
            counter['recall'][entity_type] = recall
            counter['f1'] = 2 * precision * recall / (precision + recall)
        return counter

    def __sent_label2span(self, pred_labels, true_labels):
        pred_spans = label2span(pred_labels, self.__label_schema)
        true_spans = label2span(true_labels, self.__label_schema)
        typed_pred_spans = aggregate_spans_by_type(pred_spans, self.__entity_types)
        typed_true_spans = aggregate_spans_by_type(true_spans, self.__entity_types)
        ret = {'true': typed_true_spans, 'pred': typed_pred_spans}
        return ret

    def __get_data(self, data):
        if data is None:
            return data
        elif isinstance(data, str):
            return read_json(data)
        elif isinstance(data, list):
            return data
        else:
            raise TypeError('input data type {} is invalid in evaluation'.format(type(data)))

    def __check_data(self, pred_data):
        if len(pred_data) != len(self.__true_data):
            raise LengthNotEqualException('pred data count is not equal to true data count')
        msg_tmpl = 'evaluation data error. {} is not equal to {}'

        for pred_sent, true_sent in zip(pred_data, self.__true_data):
            pred_text = pred_sent['text']
            true_text = true_sent['text']
            if pred_text != true_text:
                raise ValueError(msg_tmpl.format(pred_text, true_text))
            if len(pred_sent['labels']) != len(true_sent['labels']):
                raise ValueError('predicted labels count is not equal to true labels')


def aggregate_spans_by_type(spans, entity_types):
    typed_data = {t: [] for t in entity_types}

    for start, end, entity_type in spans:
        typed_data[entity_type].append((start, end))
    return typed_data


def aggregate_entities_in_sents_by_type(data, entity_types):
    new_data = {t: [] for t in entity_types}
    for sent in data:
        entities = sent['entities']
        for entity_type in entity_types:
            new_sent = copy.deepcopy(sent)
            new_sent['entities'] = [e for e in entities if e['type'] == entity_type]
            new_data[entity_type].append(new_sent)
    return new_data
