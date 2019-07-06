# -*- coding: UTF-8 -*-
import copy
import pytest
from datagrand_ie_2019.utils.utils import *


def test_get_index_char2word():
    tokens = [{'text': '我们', 'start': 0, 'end': 2}, {'text': '来自', 'start': 2, 'end': 4},
              {'text': '智慧芽', 'start': 4, 'end': 7}]
    assert get_index_char2word(tokens, 1) == 0
    assert get_index_char2word(tokens, 2) == 1
    assert get_index_char2word(tokens, 3) == 1
    assert get_index_char2word(tokens, 5) == 2
    assert get_index_char2word(tokens, 6) == 2
    with pytest.raises(IndexError):
        get_index_char2word(tokens, 7)


def test_replace_item_in_list():
    old_list1 = list(range(10))
    replaced_items1 = [(3, 12)]
    new_list1 = replace_item_in_list(old_list1, replaced_items1)
    true_list1 = old_list1.copy()
    true_list1[3] = 12
    assert true_list1 == new_list1

    replaced_items2 = [(3, [1, 2])]
    new_list2 = replace_item_in_list(old_list1, replaced_items2, extend_list=True)
    true_list2 = old_list1.copy()
    true_list2.pop(3)
    true_list2.insert(3, 1)
    true_list2.insert(4, 2)
    assert true_list2 == new_list2

    new_list3 = replace_item_in_list(old_list1, replaced_items2, extend_list=False)
    true_list3 = old_list1.copy()
    true_list3.pop(3)
    true_list3.insert(3, [1, 2])
    assert true_list3 == new_list3


def test_replace_extname():
    src_name1 = '/mnt/data1/a.json'
    true_name1 = '/mnt/data1/a.conll'
    assert true_name1 == replace_extname(src_name1, 'conll')
    assert true_name1 == replace_extname(src_name1, '.conll')


def test_merge_spans():
    spans1 = [{'start': 1, 'end': 10}, {'start': 2, 'end': 5}]
    assert merge_spans(spans1) == [{'start': 1, 'end': 10}]

    spans2 = [{'start': 1, 'end': 10, 'color': 'yellow'},
              {'start': 2, 'end': 11, 'color': 'blue'},
              {'start': 20, 'end': 100, 'color': 'blue'},
              {'start': 200, 'end': 300}]
    spans2_ori = copy.deepcopy(spans2)
    expected_ret2 = [{'start': 1, 'end': 11, 'color': 'blue'},
                     {'start': 20, 'end': 100, 'color': 'blue'},
                     {'start': 200, 'end': 300}]
    assert merge_spans(spans2) == expected_ret2
    # original spans not changed
    assert spans2 == spans2_ori

    spans3 = [{'start': 1, 'end': 10}, {'start': 10, 'end': 20}]
    expected_ret3_adj = [{'start': 1, 'end': 20}]
    expected_ret3_no_adj = [{'start': 1, 'end': 10}, {'start': 10, 'end': 20}]
    assert merge_spans(spans3) == expected_ret3_no_adj
    assert merge_spans(spans3, merge_adjacency=True) == expected_ret3_adj
