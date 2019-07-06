# -*- coding: UTF-8 -*-
from datagrand_ie_2019.utils.constant import (VALIDATION_FILE,
                                                                          VALIDATION_OOV_FILE, EVALUATION_DIR)
from datagrand_ie_2019.utils.visualizer import *


def test_visualize_spans():
    text = 'This is an good example.'
    spans = [{'start': 0, 'end': 4},
             {'start': 5, 'end': 7},
             {'start': 8, 'end': 10, 'color': 'blue'},
             {'start': 11, 'end': 15, 'color': 'green'},
             {'start': 16, 'end': 23}]
    hl_text = visualize_spans(text, spans)

    true_text = '[0;30;43mThis[0m [0;30;43mis[0m [0;30;44man[0m [0;37;42mgood[0m [0;30;43mexample[0m.'
    assert hl_text == true_text
    print(hl_text)


def test_visualize_spans_with_token_separator():
    text = '我们来自智慧芽'
    tokens = [{'text': '我们', 'start': 0, 'end': 2}, {'text': '来自', 'start': 2, 'end': 4},
              {'text': '智慧芽', 'start': 4, 'end': 7}]
    spans = [{'start': 0, 'end': 2}, {'start': 2, 'end': 4}, {'start': 4, 'end': 7}]
    hl_text = visualize_spans_with_token_separator(text, tokens, spans)
    assert hl_text == '[0;30;43m我们[0m [0;30;43m来自[0m [0;30;43m智慧芽[0m'


def test_ner_visualizer():
    color_mapper = {'iupac': 'yellow'}
    NerVisualizer.visualize(VALIDATION_FILE, color_mapper)
    visualizer = NerVisualizer(VALIDATION_FILE, color_mapper=color_mapper)
    visualizer.compare_model('api_model')


def test_ner_visalizer_missing():
    text = 'This is BBC, and that is CNN.'
    pred_entities1 = [{'entity': 'BBC', 'start': 8, 'end': 11, 'type': 'ORG'}]
    pred_entities2 = [{'entity': 'CNN.', 'start': 25, 'end': 29, 'type': 'ORG'}]
    true_entities = [{'entity': 'BBC', 'start': 8, 'end': 11, 'type': 'ORG'},
                     {'entity': 'CNN', 'start': 25, 'end': 28, 'type': 'ORG'}]
    pred_data1 = [{'text': text, 'entities': pred_entities1}]
    pred_data2 = [{'text': text, 'entities': pred_entities2}]
    true_data = [{'text': text, 'entities': true_entities}]
    visualizer = NerVisualizer(true_data, color_mapper={'ORG': 'yellow'})
    visualizer.compare_sents_missing(pred_data1)
